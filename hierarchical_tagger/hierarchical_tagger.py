import copy
import json
import os
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import AnyStr, Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sknetwork.clustering import Louvain
from treelib import Tree


def make_empty_array():
    return np.ndarray([])


def make_empty_csr_matrix():
    return csr_matrix([])


DataProperty = namedtuple(
    "DataProperty", ["n_terms", "n_occurrences", "core_terms", "term_set", "label"]
)


@dataclass
class HierarchicalTagger:
    """
    # Basic use:

    Data must be dictionaries of the form:
    {doc_id : ["term_1", "term_2", ...]}
    DOCS  = {"1": ["Burgers", "Vegetarian Diet"],
         "2": ["Britney Spears", "Music", "Conservatorship"]}

    # Instantiate
    hierarchical_tagger = HierarchicalTagger()

    # Ingest documents
    # Ingest can take time. See below for saving and reloading your object
    hierarchical_tagger.ingest(document_terms=DOCS)

    # Learn tag tree
    hierarchical_tagger.fit_tag_tree()

    # Inspect tag tree
    hierarchical_tagger.tree.show()   # c.tree is a treelib Tree object

    # Tag documents
    hierarchical_tagger.tag_documents()

    # Inspect document tags
    hierarchical_tagger.document_tags # {doc_id : [(tag, score, approximate hierarchy level), ...]}

    # Saving to JSON string
    serialized = hierarchical_tagger.to_json()

    # Loading from JSON string
    hierarchical_tagger = HierarchicalTagger.from_json(
        serialized,
        hydrate_tree=True,
        hydrate_tags=True
    )

    Advanced options:

    # Upon document ingestion

    term_suggestions: List[AnyStr] = A list of suggested terms to consider when building the tree

    filter_geo_terms: bool = A boolean parameter to exclude geographical terms from the term set

    document_attributes: Dict[AnyStr, Dict] = A dictionary storing additional arbitrary attributes about the documents.
    Expected schema is {doc_id: {attribute_1_name: attribute_1_value, attribute_2_name: attribute_2_value, ... }.
    This field is useful to store, for example, document title and text for use in later lookups or comparisons.
    This field is also used in the webapp.

    # Upon fitting tree

    Additional terms can be suggested into the term set (note this can only be done on document
    ingestion, see above). As the hierarchical clustering step takes account of the empirical
    document frequency in the corpus, suggested terms must also be assigned an estimate/prior of
    their likely document frequencies to help guide the breadth of the relevance of each term. In
    some cases, it may be useful to estimate these document frequencies from a separate corpus.
    -  term_suggestions_doc_freq: List[float] = A prior on doc frequency for terms in
    term_suggestions

    The user can provide a list of terms to exclude. These terms as well as their synonyms
    (precisely, all terms falling into a cluster including a term on the excluion list, see next
    paragraph) are removed from the cluster set.
    -  term_exclusion_list: List[AnyStr] = A list of suggested terms to exclude when building the
    tree. Synonyms are also dropped.

    Similar terms are deduplicated using Louvain clustering on embeddings representing the term's
    meanings.
    -  term_similarity_threshold: float [0,1] = Controls the similarity threshold when constructing
    the adjacency matrix for this clustering.

    Terms in clusters whose terms have low document frequency across all terms in the cluster are
    dropped.
    -  min_term_cluster_count: int [1,n_docs] = The minimum count of term occurrences across docs
    for all terms in each cluster.

    Tree fitting is non-deterministic. Use random_state for replicability.
    -  random_state: int = None

    # Upon tagging documents

    When tagging a document, we want to also ensure a connection is made to broader / more abstract
    domains. For example, 'batteries' could be mapped to something like:
      'battery storage' > 'electricity systems' > 'renewable energy' > 'energy' > 'sustainability'.
    The min_abstraction_similarity parameter gives control over how far removed an abstraction can
    be from the original term.
    -  min_abstraction_similarity:  float [0,1]= .2

    Each tag assigned to a document is also given a score measuring how related it is to that
    docuement. From observation, a score of >0.4 is very strong, results are quite decent until
    ~0.2 and get patchy below that. The min_tag_score parameter sets the threshold below which tags
    will not be assigned to documents.
    -  min_tag_score: float = .15
    """

    # INGEST INPUT
    document_terms: Dict[AnyStr, List] = field(default_factory=dict)
    document_attributes: Dict[AnyStr, Dict] = field(default_factory=dict)
    term_suggestions: List[AnyStr] = field(default_factory=list)
    filter_geo_terms: bool = False

    # DERIVED UPON INGEST
    n_docs: int = 0
    filtered_terms: List[AnyStr] = field(default_factory=list)
    term_pipeline: Dict[AnyStr, Dict] = field(default_factory=dict)
    term_counts: Counter = field(default_factory=Counter)
    term_doc_freq: Dict[AnyStr, float] = field(default_factory=dict)
    _term_embeddings: np.array = field(default_factory=make_empty_array)
    _filtered_term_similarity: np.array = field(default_factory=make_empty_array)

    # FIT TREE INPUT
    term_exclusion_list: List[AnyStr] = field(default_factory=list)
    term_suggestions_doc_freq: List[float] = field(default_factory=list)
    term_similarity_threshold: float = 0.9
    min_term_cluster_count: int = 2
    random_state: int = None

    # DERIVED UPON FITTING TREE
    grouped_terms: Dict = field(default_factory=dict)
    _collapsed_term_pipeline: Dict = field(default_factory=dict)
    processed_document_terms: Dict[AnyStr, List] = field(default_factory=dict)
    selected_terms: List[AnyStr] = field(default_factory=list)
    _n_selected_terms: int = 0
    _selected_terms_idxs: List = field(default_factory=list)
    selected_term_counts: Counter = field(default_factory=Counter)
    selected_term_doc_freq: Dict[AnyStr, float] = field(default_factory=dict)

    # TAG DOCUMENTS INPUT
    min_abstraction_similarity: float = 0.2
    min_tag_score: float = 0.15

    # CONSTANTS
    TERM_DROP_KEY: str = "---DROP---"
    # From https://www.sbert.net/
    ENCODER = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

    # A mapping of output format to list of attribute names for use in serialization.
    SERIALIZED_ATTRS = {
        "json_ready": [
            "document_terms",
            "document_attributes",
            "term_exclusion_list",
            "term_suggestions",
            "filter_geo_terms",
            "n_docs",
            "filtered_terms",
            "term_pipeline",
            "term_counts",
            "term_doc_freq",
            "term_suggestions_doc_freq",
            "term_similarity_threshold",
            "min_term_cluster_count",
            "random_state",
            "min_abstraction_similarity",
            "min_tag_score",
        ],
        "csr_matrix": ["_filtered_term_similarity"],
        "numpy": ["_term_embeddings"],
    }

    #
    # SERIALIZATION METHODS
    #

    @classmethod
    def from_json(cls, serialized_obj, hydrate_tree=False, hydrate_tags=False):

        obj_dict = json.loads(serialized_obj)
        return cls.from_dict(
            obj_dict, hydrate_tree=hydrate_tree, hydrate_tags=hydrate_tags
        )

    @classmethod
    def from_dict(cls, obj_dict, hydrate_tree=False, hydrate_tags=False):

        # Convert attributes to target format as per SERIALIZED_ATTRS.
        for attr_group, attrs in cls.SERIALIZED_ATTRS.items():
            if attr_group == "numpy":
                for attr in attrs:
                    obj_dict[attr] = np.array(obj_dict[attr])
            elif attr_group == "csr_matrix":
                for attr in attrs:
                    # csr_matrix is used for serialization, but obj is transformed to dense array
                    # for calculations
                    obj_dict[attr] = csr_matrix(
                        (
                            obj_dict[attr]["data"],
                            (obj_dict[attr]["row"], obj_dict[attr]["col"]),
                        ),
                        shape=obj_dict[attr]["shape"],
                    ).toarray()
            elif attr_group == "set":
                for attr in attrs:
                    obj_dict[attr] = set(obj_dict[attr])
            else:
                # obj_dict[attr] is already of expected type
                continue

        ht_instance = cls(**obj_dict)

        if any([hydrate_tree, hydrate_tags]):
            # Recalculate tree with parameter values from serialized object
            ht_instance.fit_tag_tree(
                **{
                    attr: value
                    for attr, value in obj_dict.items()
                    if attr
                    in [
                        "term_suggestions_doc_freq",
                        "term_similarity_threshold",
                        "min_term_cluster_count",
                        "random_state",
                    ]
                }
            )

        if hydrate_tags:
            # Recalculate document tags with parameter values from serialized object
            ht_instance.tag_documents(
                **{
                    attr: value
                    for attr, value in obj_dict.items()
                    if attr in ["min_abstraction_similarity", "min_tag_score"]
                }
            )

        return ht_instance

    def to_json(self):

        obj_dict = {
            attr: getattr(self, attr)
            for attrs in self.SERIALIZED_ATTRS.values()
            for attr in attrs
        }

        # Use smaller csr_matrix for serialization
        obj_dict["_filtered_term_similarity"] = csr_matrix(
            obj_dict["_filtered_term_similarity"]
        )

        return json.dumps(obj_dict, cls=CustomEncoder)

    def ingest(
        self,
        document_terms: Dict[AnyStr, List],
        document_attributes: Dict[AnyStr, Dict] = None,
        term_suggestions: List[AnyStr] = None,
        filter_geo_terms: bool = False,
        term_similarity_minimum: float = 0.6,
    ):
        """
        The ingest step involves going from the terms as they appear in the documents, to a filtered
        set of terms to use as candiates in the successive tree fitting step.

        We attempt to reduces the term set by combining plurals with singulars and, optionally,
        removing geographical names. We then map all the remining filtered_terms into an embedding
        space using a large language model.
        """

        # Set up starting values
        self._intialize_ingest(document_terms, document_attributes, term_suggestions)

        # Create a clean term set from documents and suggestions
        self._clean_up_terms(
            filter_geo_terms, term_similarity_minimum=term_similarity_minimum
        )

    def _intialize_ingest(self, document_terms, document_attributes, term_suggestions):
        # Resets attributes to empty containers and/or user arguments.
        self.term_pipeline = dict()
        self.filtered_terms = set()
        self.grouped_terms = dict()
        self.selected_terms = list()
        self.term_suggestions = (
            [t.lower() for t in term_suggestions]
            if term_suggestions is not None
            else list()
        )
        self.document_terms = document_terms
        self.document_attributes = document_attributes
        self.n_docs = len(document_terms)

    def _clean_up_terms(self, filter_geo_terms, term_similarity_minimum=0.6):
        # Create term set from documents and suggestions
        self._create_term_set()
        # Remove plurals
        self._remove_plurals()
        # Exclude geographical terms
        self.filter_geo_terms = filter_geo_terms
        if self.filter_geo_terms is True:
            self._apply_geo_term_filter()
        # Map terms to embedding space
        self._semantic_term_map()
        # Create a term similarity matrix, subject to minimum similarity
        self._create_filtered_term_similarity(
            term_similarity_minimum=term_similarity_minimum
        )

    def _create_term_set(self):
        # Creates initial set of all terms. Also makes lowercase.
        lowercase_terms = [
            term.lower()
            for document in self.document_terms.values()
            for term in document
        ]
        self.filtered_terms = set(lowercase_terms)

        # Add suggestions to filtered_terms set
        self.filtered_terms = self.filtered_terms.union(set(self.term_suggestions))

        # Term counts and frequencies from docs
        self.term_counts = Counter(lowercase_terms)
        # Calculate term document frequencies.
        # Assumes that a term is not present twice in any document.
        self.term_doc_freq = {
            term: self.term_counts[term] / self.n_docs for term in self.filtered_terms
        }

    def _remove_plurals(self):
        # Creates a term_pipeline step mapping plural terms to their singular
        plurals = set()
        remove_plurals_step = dict()
        for term in self.filtered_terms:
            # For each plural term where singular is also in set
            # Drop the plural and move its counts to the singular
            if term[-1] == "s" and term[0:-1] in self.filtered_terms:
                plurals.add(term)
                remove_plurals_step[term] = term[0:-1]
                self.term_counts[term[0:-1]] += self.term_counts[term]
                self.term_counts[term] = 0
            if term[-3:] == "ies" and term[0:-3] + "y" in self.filtered_terms:
                plurals.add(term)
                remove_plurals_step[term] = term[0:-3] + "y"
                self.term_counts[term[0:-3] + "y"] += self.term_counts[term]
                self.term_counts[term] = 0

        self.term_pipeline["1: Remove Plurals"] = remove_plurals_step
        self.filtered_terms = self.filtered_terms - plurals

    def _apply_geo_term_filter(self):
        # Creates a term_pipeline step to remove geographical locations from the term set

        # Load set of locations
        locations = get_locations_set()

        # Initialize pipeline step
        remove_geo_terms_step = dict()

        # Remove all terms that are exact matches to the location set
        geo_terms = set(self.filtered_terms).intersection(locations)
        for term in geo_terms:
            remove_geo_terms_step[term] = self.TERM_DROP_KEY
        self.filtered_terms = self.filtered_terms - geo_terms

        # Remove terms that contain locations
        def _is_sublist(sub_lst, lst):
            n = len(sub_lst)
            return any((sub_lst == lst[i : i + n]) for i in range(len(lst) - n + 1))

        for term in self.filtered_terms:
            for location in locations:
                if _is_sublist(location.split(), term.split()):
                    # Remove if location is a subsequence of words in term
                    remove_geo_terms_step[term] = self.TERM_DROP_KEY
                    geo_terms.add(term)
                    # Break out of inner loop to move to next term
                    break

        self.term_pipeline["2: Remove Geo Terms"] = remove_geo_terms_step
        self.filtered_terms = self.filtered_terms - geo_terms

    def _semantic_term_map(self):
        # Calculate embeddings for terms that passed initial filtering
        self.filtered_terms = list(self.filtered_terms)
        self._term_embeddings = self.ENCODER.encode(self.filtered_terms)

    def _create_filtered_term_similarity(self, term_similarity_minimum=0.6):
        self._filtered_term_similarity = cosine_similarity(self._term_embeddings)
        # To reduce memory footprint we set all 'low' similarities to zero and convert to csr_matrix
        # upon serialization
        if term_similarity_minimum:
            self._filtered_term_similarity[
                self._filtered_term_similarity < term_similarity_minimum
            ] = 0

    #
    # EXTRACT THE HIERARCHICAL TAG TREE
    #

    def fit_tag_tree(
        self,
        term_exclusion_list: List[AnyStr] = None,
        term_suggestions_doc_freq: List[float] = None,
        term_similarity_threshold: float = 0.9,
        min_term_cluster_count: int = 2,
        random_state: float = None,
    ):
        """
        This step fits the tag tree from the document terms.

        Some pre-processing steps are carried out further reduce the candidate term sets. We group
        semantically similar terms. Terms in groups with low overall document count are dropped.
        Terms included in the user-provided exclusion list, as well as their synonyms, are also
        dropped. Finally, we select a single term from each remaining group. Terms passing this
        processing step are referred to as selected_terms, because they are selected as candidate
        terms to appear in the hierarchical tree.
        """

        self._initialize_fit(
            term_exclusion_list,
            term_suggestions_doc_freq,
            term_similarity_threshold,
            min_term_cluster_count,
            random_state,
        )

        self._process_terms()

        self._extract_tag_tree()

    def _initialize_fit(
        self,
        term_exclusion_list,
        term_suggestions_doc_freq,
        term_similarity_threshold,
        min_term_cluster_count,
        random_state,
    ):
        """Resets attributes to empty containers and/or user arguments."""

        self.term_exclusion_list = (
            list({t.lower() for t in term_exclusion_list})
            if term_exclusion_list is not None
            else list()
        )
        self.term_similarity_threshold = term_similarity_threshold
        self.min_term_cluster_count = min_term_cluster_count
        self.term_suggestions_doc_freq = (
            term_suggestions_doc_freq
            if term_suggestions_doc_freq is not None
            else list()
        )
        if len(self.term_suggestions) != len(self.term_suggestions_doc_freq):
            raise ValueError(
                "term_suggestions and term_suggestions_doc_freq must be of equal length."
            )
        self.random_state = (
            random_state if random_state is not None else np.random.randint(5000)
        )

    def _process_terms(self):
        # Ingest term suggestion document frequencies
        self._read_term_suggestion_doc_freq()
        # Group semantically similar terms
        self._group_terms()
        # Drop low count term clusters
        self._drop_low_count_term_clusters()
        # Apply term exclusion list
        self._apply_term_exclusion_list()
        # Select representative terms
        self._select_representative_terms()
        # Run pipeline to process documents and select final term set
        self._run_term_pipeline()
        # Refocus internals on selected terms only
        self._set_up_for_selected_terms()

    def _read_term_suggestion_doc_freq(self):
        """Add or overwrite doc_freq and counts for suggestions"""
        for suggestion, suggestion_doc_freq in zip(
            self.term_suggestions, self.term_suggestions_doc_freq
        ):
            self.term_doc_freq[suggestion] = suggestion_doc_freq
            self.term_counts[suggestion] = int(
                self.term_doc_freq[suggestion] * self.n_docs
            )

    def _group_terms(self):
        """
        Cluster semantically similar terms using Louvain clustering.
        In this step, we are aiming to find synonyms, so term_similarity_threshold should be high
        (ex. >=0.9)
        """

        louvain = Louvain(random_state=self.random_state)
        adjacency = copy.deepcopy(self._filtered_term_similarity)
        adjacency[adjacency < self.term_similarity_threshold] = 0
        adjacency = csr_matrix(adjacency)
        labels = louvain.fit_transform(adjacency)

        # Store clustering in dict mapping cluster_label to list of terms
        grouped_terms = defaultdict(list)
        for t, l in zip(self.filtered_terms, labels):
            grouped_terms[str(l)].append(t)

        self.grouped_terms = grouped_terms

    def _drop_low_count_term_clusters(self):
        """Creates a term_pipeline step to drop all terms in clusters with low overall term count"""

        low_term_count_clusters = [
            k
            for k, v in self.grouped_terms.items()
            if sum(self.term_counts[t] for t in v) < self.min_term_cluster_count
        ]

        low_term_count_step = {
            term: self.TERM_DROP_KEY
            for cluster_id, terms in self.grouped_terms.items()
            for term in terms
            if cluster_id in low_term_count_clusters
        }
        self.term_pipeline["3: Remove Low Count Terms"] = low_term_count_step

        self.grouped_terms = {
            k: v
            for k, v in self.grouped_terms.items()
            if k not in low_term_count_clusters
        }

    def _apply_term_exclusion_list(self):
        """
        Creates a term_pipeline step to drop clusters containing terms in the exclusion list
        This means that synonyms of the excluded term will also be dropped.
        The exclusion list should therefore list *concepts* not merely specific variants of a term.
        """

        exclusion_clusters = [
            cluster_id
            for cluster_id, terms in self.grouped_terms.items()
            if any((term in self.term_exclusion_list) for term in terms)
        ]

        exclusion_step = {
            term: self.TERM_DROP_KEY
            for cluster_id, terms in self.grouped_terms.items()
            for term in terms
            if cluster_id in exclusion_clusters
        }

        self.term_pipeline["4: Apply Term Exclusion List"] = exclusion_step

        self.grouped_terms = {
            k: v for k, v in self.grouped_terms.items() if k not in exclusion_clusters
        }

    def _select_representative_terms(self):
        """
        Creates a term_pipeline step selecting a representative term for each cluster
        The term with the highest average cosine similarity with all other terms in the cluster is
        chosen.
        """

        term_to_representative_term_step = {}
        selected_terms = []

        for terms in self.grouped_terms.values():
            if len(terms) > 1:
                term_ids = [self.filtered_terms.index(t) for t in terms]
                # Numerical errors were leading to non-reproducibility of results.
                # Use slower dtype=float64 as in Notes here
                # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
                central_term_id = term_ids[
                    np.argsort(
                        -self._filtered_term_similarity[term_ids, :][:, term_ids].sum(
                            axis=1, dtype="float64"
                        )
                    )[0]
                ]
                central_term = self.filtered_terms[central_term_id]
                selected_terms.append(central_term)
                for term in terms:
                    if term != central_term:
                        term_to_representative_term_step[term] = central_term

        self.term_pipeline[
            "5: Select Representative Terms"
        ] = term_to_representative_term_step

    def _run_term_pipeline(self):
        """
        Having logged all term transformation steps in .term_pipeline, we now run the original
        document_terms through the pipeline to generate processed_document_terms: a document
        representation using terms that have been fully processed.
        """

        # Collapse all pipeline steps to generate a single DAG showing how each term is transformed
        # at each step.
        self._collapsed_term_pipeline = dict()
        for pipeline_step in self.term_pipeline.values():
            self._collapsed_term_pipeline.update(pipeline_step)

        # Run all document terms through term pipeline
        processed_document_terms = {}
        for document, term_list in self.document_terms.items():
            processed_terms = []
            for t in term_list:
                # Run lowercase term through pipeline
                processed_term = next(self._run_term_through_pipeline(t.lower()))
                if processed_term is not None:
                    processed_terms.append(processed_term)

            processed_document_terms[document] = list(set(processed_terms))

        self.processed_document_terms = processed_document_terms

    def _run_term_through_pipeline(self, term):
        """
        Recursively traverse the term transformation DAG and yield the final term, or None if term
        is dropped.
        """
        term_result = self._collapsed_term_pipeline.get(term)
        if term_result is None:
            # No further pipeline steps
            yield term
        if term_result == self.TERM_DROP_KEY:
            # Term should be dropped
            yield None
        # Continue pipeline
        yield from self._run_term_through_pipeline(term_result)

    def _set_up_for_selected_terms(self):
        """Set up the final selected_terms and associated embeddings and similarity matrix"""

        # Create set of selected terms from documents after term processing
        doc_selected_terms = [
            term
            for doc_terms in self.processed_document_terms.values()
            for term in doc_terms
        ]

        # Process suggestion terms, and keep only the ones that pass all filters
        processed_suggested_terms = [
            next(self._run_term_through_pipeline(t)) for t in self.term_suggestions
        ]
        processed_suggested_terms = [
            t for t in processed_suggested_terms if t is not None
        ]
        suggested_selected_terms = set(self.filtered_terms).intersection(
            set(processed_suggested_terms)
        )

        # Combine into final selected_terms
        self.selected_terms = list(
            set(doc_selected_terms).union(suggested_selected_terms)
        )
        self._n_selected_terms = len(self.selected_terms)

        # Overwrite count data considering only selected term
        self.selected_term_counts = Counter(doc_selected_terms)
        self.selected_term_doc_freq = {
            term: self.selected_term_counts[term] / self.n_docs
            for term in set(doc_selected_terms)
        }

        # Add or overwrite doc_freq for suggestions
        for suggestion, suggestion_doc_freq in zip(
            self.term_suggestions, self.term_suggestions_doc_freq
        ):
            if suggestion in suggested_selected_terms:
                self.selected_term_doc_freq[suggestion] = suggestion_doc_freq
                self.selected_term_counts[suggestion] = int(
                    self.selected_term_doc_freq[suggestion] * self.n_docs
                )

        # Slice embeddings matrix to focus on selected terms
        self._selected_terms_idxs = [
            self.filtered_terms.index(t) for t in self.selected_terms
        ]
        self._selected_terms_embeddings = copy.deepcopy(
            self._term_embeddings[self._selected_terms_idxs]
        )

        # Calculate similarity across selected terms
        self._selected_terms_similarity = cosine_similarity(
            self._selected_terms_embeddings
        )

    def _extract_tag_tree(self):

        # Run hierarchical clustering on terms
        self._build_term_hierarchy()

        # Build tree from hierarchy
        self._build_tag_tree()

    def _build_term_hierarchy(self):
        self.hierarchy = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            memory=None,
            connectivity=None,
            compute_full_tree="auto",
            linkage="average",
            distance_threshold=0,
            compute_distances=False,
        )
        self.hierarchy.fit(self._selected_terms_embeddings)

    def _build_tag_tree(self):
        """
        Build tree downwards starting from highest node
        AgglomerativeClustering.hierarchy.children_ stores the ways in which terms are aggregated
        hierarchically we build a tree from these relationships, adding some custom logic and
        descriptive data in the process.
        """

        self.tree = Tree()
        node_id = self.hierarchy.children_.max() + 1
        self._build_term_tree_from_node(
            node_id, parent_node_id=None, parent_node_term=None
        )

        # Filter to core nodes, ie ensuring node labels are unique
        self._define_core_nodes()

        # Build tag tree from core nodes only
        self._build_pruned_tree()

        # Replace tree with pruned_tree
        self.tree = copy.deepcopy(self.pruned_tree)
        self.pruned_tree = None

    def _build_term_tree_from_node(self, node_id, parent_node_id, parent_node_term):
        """
        Starting from node_id, recursively traverses all downstream nodes and leaves of
        self.hierarchy. At each step computes information about the node/leave in terms of the
        downstream nodes and, importantly, selects the most central term to be the name of this
        node.
        """

        # Find all terms that contained in this node (i.e. are downstream leaves)
        term_ids = list()
        self._find_node_term_ids(term_ids, node_id)

        # Number of terms and their occurrences in the documents
        n_terms = len(term_ids)
        n_occurrences = sum(self.term_counts[self.selected_terms[i]] for i in term_ids)

        # Represent the node via its contained terms, centrality ranking, and scores
        ranked_term_ids, ranked_term_id_scores = self._extract_ranked_term_ids(term_ids)
        ranked_terms = [self.selected_terms[i] for i in ranked_term_ids]
        terms_set = [
            (term, score) for term, score in zip(ranked_terms, ranked_term_id_scores)
        ]

        # Find all downstream elements
        leaves, nodes = self._find_node_children(node_id)

        if ranked_terms[0] != parent_node_term:
            # If central term for this node different from parent node
            # Create a new node with associated term and other metadata
            node_term = ranked_terms[0]
            self.tree.create_node(
                node_term,
                node_id,
                parent=parent_node_id,
                data=DataProperty(
                    n_terms,
                    n_occurrences,
                    ranked_terms[:3],
                    terms_set,
                    f"{node_term} - {n_occurrences}",
                ),
            )
        else:
            # If central term for this node is the same a parent node
            # Don't create new node and pass parent node id and term downstream
            node_id = parent_node_id
            node_term = parent_node_term

        # Proceed down into the tree, handling nodes and leaves differently

        # Recursively move to the downstream nodes, passing this node as parent
        for node in nodes:
            self._build_term_tree_from_node(node, node_id, node_term)

        # Create nodes directly for all immediate downstream leaves
        for leaf in leaves:
            leaf_term = self.selected_terms[leaf]
            if leaf_term != node_term:
                # Create leaf nodes if leaf is a different term
                self.tree.create_node(
                    leaf_term,
                    leaf,
                    parent=node_id,
                    data=DataProperty(
                        1,
                        self.term_counts[leaf_term],
                        [leaf_term],
                        [leaf_term],
                        f"{leaf_term} - {self.term_counts[leaf_term]}",
                    ),
                )

    def _find_node_children(self, node_id):
        """
        Find downstream nodes and leaves for a given node_id
        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        Section: Attributes -> children_
        """
        if node_id < self._n_selected_terms:
            # Leaf node
            return [node_id], []
        children = self.hierarchy.children_[node_id - self._n_selected_terms]
        leaves = [child for child in children if child < self._n_selected_terms]
        nodes = [child for child in children if child >= self._n_selected_terms]
        return leaves, nodes

    def _find_node_term_ids(self, term_ids, node_id):
        """
        Recursively traverse to tree down to the leaves, collecting term_ids for all downstream
        terms.
        """
        leaves, nodes = self._find_node_children(node_id)
        term_ids.extend(leaves)
        for node in nodes:
            self._find_node_term_ids(term_ids, node)

    def _extract_ranked_term_ids(self, term_ids):
        """
        Rank selected term_ids by (descending) measure of centrality
        Centrality is measured by the average term similarity across all terms in the set
        weighted by the term document frequencies: semantic similarity to a frequent term will count
        more.
        """

        # Similarities in the term set
        sim_subgraph = self._selected_terms_similarity[term_ids, :][:, term_ids]
        # Term document frequencies
        weights = np.array(
            [self.selected_term_doc_freq[self.selected_terms[i]] for i in term_ids]
        )
        # Weighted average term similarity
        term_scores = (sim_subgraph * weights.T).sum(axis=1)

        # Rankings
        ranked_term_ids = [term_ids[i] for i in np.argsort(-term_scores)]
        ranked_term_id_scores = -np.sort(-term_scores)
        return ranked_term_ids, ranked_term_id_scores

    def _define_core_nodes(self, min_n_terms=1):
        """
        Defines set of core_nodes.
        If two nodes have the same central term (node.tag), choose the node with the higher document
        frequency.
        Can also limit nodes to those containing a min_n_terms.
        """

        descending_nodes = sorted(
            self.tree.all_nodes(), key=lambda x: x.data.n_occurrences, reverse=True
        )

        used_terms = set()
        core_nodes = []
        for node in descending_nodes:
            if node.tag not in used_terms and node.data.n_terms > min_n_terms:
                core_nodes.append(node)
                used_terms.add(node.tag)

        self._core_nodes = core_nodes

    def _build_pruned_tree(self):
        """Builds a tree that only exists of nodes in the core nodes"""
        core_node_ids = {x.identifier for x in self._core_nodes}
        self.pruned_tree = Tree()
        for node in self._core_nodes:
            found_parent = False
            # Documentation for .predecessor does not exist (yet?)
            # See: https://github.com/caesar0301/treelib/blob/master/treelib/node.py#L129
            # See: https://github.com/caesar0301/treelib/issues/158
            old_tree_parent = node.predecessor(self.tree.identifier)
            if not old_tree_parent:
                # No parent found -> Add as root node
                self.pruned_tree.add_node(node)
                found_parent = True
            while not found_parent:
                # Look until we find ancestor in set of core nodes, and add to pruned_tree.
                if old_tree_parent in core_node_ids:
                    self.pruned_tree.add_node(node, parent=old_tree_parent)
                    found_parent = True
                else:
                    not_found_node = self.tree.nodes[old_tree_parent]
                    old_tree_parent = not_found_node.predecessor(self.tree.identifier)
        return self.pruned_tree

    #
    # TAG DOCUMENTS
    #

    def tag_documents(
        self, min_abstraction_similarity: float = 0.2, min_tag_score: float = 0.15
    ):
        """
        Maps document topics to hierarchical tags.

        We have a set of candidate tags, represented as nodes in the hierarchical tree, and stored
        in core_nodes.

        Each node has a unique label (node.tag) and its meaning is represented as a weighted
        combination of the children terms (node.data.term_set).

        We want to match each of the document topics to the closest candidate node.
        Ex 'batteries' -> 'battery storage'.
        Additioanlly, we want to also ensure a connection is made to broader / more abstract
        domains.
        Ex 'batteries' -> 'battery storage' > 'electricity systems' > 'renewable energy' >
        'energy' > 'sustainability', ideally with a declining relatedness score as we move further
        up the abstractions.

        We carry out the following steps to achieve this:
            1. Convert core_nodes into a node X term matrix representation
            2. Calculate semantic similarity scores across nodes (accounting for the semantic
                similarity across terms).
            3. Find each node's 'abstractions'. For a node J, 'abstractions' are defined as other
                nodes K appearing higher up in the tree, weighted by the similarity between J and K
                to capture the increasing remoteness of the abstraction.
            4. For each document topic, we map a) closest node (step 1) and b) the node's
                abstractions (step 3).
            5. We aggregate the topic to abstraction mapping a document level. This yields a
                document X node matrix.
            6. To account for the fact that higher level abstractions are going to show up more
                often, we run this final matrix through a tfidf transformation.
        """
        # Set values
        self.min_abstraction_similarity = min_abstraction_similarity
        self.min_tag_score = min_tag_score

        # Set up a node x term matrix
        self._build_node_term_matrix()  # Step 1; nodes X terms matrix
        self._build_node_similarity_matrix()  # Step 2; nodes X nodes matrix
        self._build_node_abstraction_matrix()  # Step 3 node X nodes matrix
        self._match_terms_to_nodes()  # Step 4a; dict term -> closest node

        # Match all documents to tags and abstractions
        self.document_tag_matrix = np.vstack(
            [
                self._document_topics_to_tags(document_terms)  # Step 4a, 4b and 5
                for document_terms in self.processed_document_terms.values()
            ]
        )  # documents x nodes matrix

        tfidf = TfidfTransformer()  # Step 6
        self.document_tags_tfidf = tfidf.fit_transform(self.document_tag_matrix)

        self.document_tags = defaultdict(list)
        doc_ids = list(self.processed_document_terms.keys())

        # Convert documents x nodes matrix to document_id -> tags dictionary
        cx = self.document_tags_tfidf.tocoo()
        for idx, i, v in zip(cx.row, cx.col, cx.data):
            if v > self.min_tag_score:
                self.document_tags[doc_ids[idx]].append(
                    tuple([self._core_nodes[i].tag, v, self._core_nodes[i].identifier])
                )

        # Sort document tags by relevance score
        for doc_id, tags in self.document_tags.items():
            self.document_tags[doc_id] = sorted(tags, key=lambda x: x[1], reverse=True)

    def _build_node_term_matrix(self):
        """
        Build a matrix mapping core nodes (rows) to their contained terms (columns), and their
        measure of centrality.
        """

        # Populate as sparse matrix
        rows = list()
        cols = list()
        data = list()
        for i, core_node in enumerate(self._core_nodes):
            row = np.array(
                [i] * core_node.data.n_terms
            )  # Rows in same order as _core_nodes
            terms, datum = zip(*core_node.data.term_set)
            col = np.array([self.selected_terms.index(term) for term in terms])
            rows.extend(row)
            cols.extend(col)
            data.extend(datum)

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)

        node_term_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self._core_nodes), len(self.selected_terms)),
        )
        node_term_matrix = normalize(node_term_matrix)

        self._node_term_matrix = node_term_matrix

        # Mapping of node identifier to node index in node_term_matrix rows
        self._node_id_to_idx = {
            core_node.identifier: i for i, core_node in enumerate(self._core_nodes)
        }
        self._node_idx_to_id = {
            i: core_node.identifier for i, core_node in enumerate(self._core_nodes)
        }

    def _build_node_similarity_matrix(self):

        """
        Calculate a matrix of cosine similarity across all nodes, but weighted to allow similar
        terms to count, at least partially, towards the similarity.

        See https://en.wikipedia.org/wiki/Cosine_similarity#Soft_cosine_measure

        Two docs ["electric cars", "urban mobility"] & ["electric vehicles", "urban mobility"]
        should have a similarity of almost 1 and not of just 0.5 as would be the case if we only
        consider the exact terms.
        To achieve this, we weigh the contribution of the "electric cars" <-> "electric vehicles"
        by their term as estimates by based on their language model embedding vectors.

        Normally we would construct a document pairwise similarity matrix using cosine similarity.
        For unit vectors, this is just the dot product of all vector pairs:

        raw_similarity_matrix = entity_set_A_features * entity_set_B_features.T

        However, we want for similar terms (electric car vs elecrtic vehicles) to count towards the
        similarity. We can do this by weighing the product by the similarity across terms.
        So we do this instead:
        raw_similarity_matrix = entity_set_A_features * term_similarity * entity_set_B_features.T

        However, this similarity score will not be bounded to 1. It will increase arbitrarily with
        the number of terms present in a document, and will include some double counting if a
        document itself is tagged with two similar terms, which then will be 'doubly' similar when
        matched to the terms in the other document.

        To remove this effect we aim to normalize similarity by the geometric mean of the
        non-normalized similarity scores of entities in set A and set B with themselves.

        We take diagonal from the non-normalized similarity matrices of sets A and B and take the
        product of all pairs. Then take the sqrt to get geometric mean.

        non_normalized_set_A_similarity_matrix =
            entity_set_A_features * term_similarity * entity_set_A_features.T

        non_normalized_set_B_similarity_matrix =
            entity_set_B_features * term_similarity * entity_set_B_features.T

        normalization_denominator =
            np.sqrt(
                non_normalized_set_A_similarity_matrix.diagonal()[:,None] *
                non_normalized_set_B_similarity_matrix.diagonal()
            )

        We then divide the raw_similarity_matrix by the normalization_denominator to get a
        similarity_matrix (approximately) bounded by 0 and 1.
        Note: Approximately as in practice we've seen values bounded by 1 but going slightly below
        0, although we have not investigated why.

        The normalization_denominator may have some 0 entries (unless all terms are used) as some
        documents may have 0 for all features. These docs will have 0 self-similarity and we get
        division by zero. This is not an issue as the numerator is bound to be 0 too, so we just
        skip these cells in the division.

        similarity_matrix[normalization_denominator>0] = (
            raw_similarity_matrix[normalization_denominator>0] /
            normalization_denominator[normalization_denominator>0]
        )
        """

        raw_node_sim = (
            self._node_term_matrix
            * self._selected_terms_similarity
            * self._node_term_matrix.T
        )
        normalization_denominator = np.sqrt(
            raw_node_sim.diagonal()[:, None] * raw_node_sim.diagonal()
        )
        node_similarity_matrix = copy.deepcopy(raw_node_sim)
        node_similarity_matrix[normalization_denominator > 0] = (
            raw_node_sim[normalization_denominator > 0]
            / normalization_denominator[normalization_denominator > 0]
        )

        self._node_similarity_matrix = node_similarity_matrix

    def _build_node_abstraction_matrix(self):
        """Matrix mapping nodes (rows) to all related higher-order nodes (ie abstractions)."""

        rows = list()
        cols = list()
        data = list()
        for i in range(len(self._core_nodes)):
            _, col, scores = zip(*self._get_node_abstractions(i))
            row = np.array([i] * len(scores))
            rows.extend(row)
            cols.extend(col)
            data.extend(scores)

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)

        node_abstraction_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(self._core_nodes), len(self._core_nodes))
        )

        self._node_abstraction_matrix = node_abstraction_matrix

    def _get_node_abstractions(self, node_idx):
        """
        Look up all higher-order nodes for given node_id and collect node.identifier and similarity
        score.
        """
        lineage = self._show_term_lineage(self._core_nodes[node_idx].identifier)

        abstractions = [
            (
                l.tag,
                self._node_id_to_idx[l.identifier],
                self._node_similarity_matrix[
                    node_idx, self._node_id_to_idx[l.identifier]
                ],
            )
            for l in lineage
            if self._node_id_to_idx.get(l.identifier) is not None
        ]

        return abstractions

    def _show_term_lineage(self, node_id):
        # Return all the parent nodes up to the top

        lineage = [self.tree.nodes[node_id]]

        while self.tree.parent(node_id) is not None:
            lineage.append(self.tree.parent(node_id))
            node_id = self.tree.parent(node_id).identifier

        lineage.reverse()
        return lineage

    def _match_terms_to_nodes(self):
        # Refer to discussion in _build_node_similarity_matrix
        # Here we are matching the terms themselves to the nodes. The matrix in the left of
        # selected_term_similarity would just be the identity matrix.
        raw_term_to_node_sim = (
            self._selected_terms_similarity * self._node_term_matrix.T
        )

        # non_normalized_single_term_similarity_matrix =
        # identity x selected_term_similarity x identity -->
        # selected_term_similarity (diagonal is one...)
        non_normalized_node_similarity_matrix = (
            self._node_term_matrix
            * self._selected_terms_similarity
            * self._node_term_matrix.T
        )
        normalization_denominator = np.sqrt(
            self._selected_terms_similarity.diagonal()[:, None]
            * non_normalized_node_similarity_matrix.diagonal()
        )
        term_to_node_sim = copy.deepcopy(raw_term_to_node_sim)
        term_to_node_sim[normalization_denominator > 0] = (
            raw_term_to_node_sim[normalization_denominator > 0]
            / normalization_denominator[normalization_denominator > 0]
        )
        # Closest node to term. Matrix to dictionary.
        term_to_node_matches = np.argmax(term_to_node_sim, axis=1)
        self._term_to_node_idx = {
            term: node_idx
            for term, node_idx in zip(self.selected_terms, term_to_node_matches)
        }

    def _document_topics_to_tags(self, document_terms):
        # For all document terms find matching closest node ids
        node_ids = [self._term_to_node_idx[term] for term in document_terms]
        # Look up nodes and their abstractions in node_abstraction_matrix
        node_abstractions = self._node_abstraction_matrix[node_ids]
        # Silence all abstractions below threshold similarity value
        node_abstractions[node_abstractions <= self.min_abstraction_similarity] = 0
        # Sum across to get document to tags match with weights
        return node_abstractions.sum(axis=0)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, csr_matrix):
            obj_coo = obj.tocoo()
            return {
                "data": obj_coo.data,
                "row": obj_coo.row,
                "col": obj_coo.col,
                "shape": obj_coo.shape,
            }
        elif isinstance(obj, set):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def get_locations_set():
    """
    Collects a standard set of locations to allow filtering these from the term set.
    This is useful if clustering documents around location is not of interest.
    As the step of filtering location names is computationally demanding,
    we restrict the set to countries, capitals, and US states and cites
    """

    # Continents, counties and capitals
    # Reference: https://gist.github.com/pamelafox/986163
    file_path = os.path.join(os.path.dirname(__file__), "data/countries.json")
    with open(file_path) as f:
        countries = json.load(f)

    continents = set()
    country_names = set()
    capitals = set()
    for country in countries:
        continents.add(country["continent"].lower())
        country_names.add(country["name"].lower())
        capitals.add(country["capital"].lower())

    # US states
    file_path = os.path.join(os.path.dirname(__file__), "data/us_states.json")
    with open(file_path) as f:
        state_names = json.load(f)
    state_names = {s.lower() for s in state_names}

    # Cities list; limit to US to restrict the set
    # Reference: https://datahub.io/core/world-cities, which itself sources the data from https://www.geonames.org/.
    file_path = os.path.join(os.path.dirname(__file__), "data/world-cities.json")
    with open(file_path) as f:
        world_cities = json.load(f)

    us_cities_list = {
        c["name"] for c in world_cities if c["country"] == "United States"
    }

    locations = continents | country_names | capitals | state_names | us_cities_list

    return locations
