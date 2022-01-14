import streamlit as st
import copy
import numpy as np

import os
from hierarchical_tagger.hierarchical_tagger import HierarchicalTagger
from webapp.utils import build_sunburst

st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def load_instance(dataset_name):
    """Loads HierarchicalTagger instance from serialized json data"""
    with open(f"./webapp/data/{dataset_name}", "r") as f:
        reloaded_serialized = f.read()
    hydrated = HierarchicalTagger.from_json(
        reloaded_serialized, hydrate_tree=False, hydrate_tags=False
    )
    return hydrated


@st.cache()
def build_tag_tree(
    hierarchical_tagger, min_term_cluster_count, term_similarity_threshold
):
    """Fits tag tree"""
    hierarchical_tagger.fit_tag_tree(
        min_term_cluster_count=min_term_cluster_count,
        term_similarity_threshold=term_similarity_threshold,
    )
    return hierarchical_tagger


@st.cache(hash_funcs={HierarchicalTagger: id})
def tag_documents(hierarchical_tagger, min_abstraction_similarity, min_tag_score):
    """
    Tag documents. Assumes .fit_tag_tree() has already been called on the HierarchicalTagger
    instance
    """
    hierarchical_tagger.tag_documents(
        min_abstraction_similarity=min_abstraction_similarity,
        min_tag_score=min_tag_score,
    )
    return hierarchical_tagger


def get_top_documents(ht, node_idx, n=5):
    doc_ids = list(ht.processed_document_terms.keys())
    return {
        doc_ids[i]: ht.document_terms[doc_ids[i]]
        for i in np.argsort(-ht.document_tags_tfidf[:, node_idx].A.ravel())[:n]
        if ht.document_tags_tfidf[i, node_idx]>0
    }


def create_sidebar(sidebar):
    sidebar.header("Datasets")
    # Expose in a dropdown any json files in the ./webapp/data/ folder
    options = ["None"] + [
        file for file in os.listdir("./webapp/data/") if file.endswith(".json")
    ]
    dataset_name = sidebar.selectbox("Choose a dataset", options, index=0)
    sidebar.header("View")
    view_options = ["Tag Tree", "Document Search"]
    view = sidebar.selectbox("Choose a view", view_options, index=0)
    return dataset_name, view


def add_sidebar_options(sidebar, include_tagging_options=False):

    tooltips = {
        "min_term_cluster_count": "Minimum number of terms needed to constitute a leaf in the chart. Increase to prune the smaller leaves and vice-versa.",
        "term_similarity_threshold": "Minimum semanitic similarity for terms to be combined into one. Increase to push the tree to split branches more easily and vice-versa.",
        "min_abstraction_similarity": "Increase if you notice that tags, especially the broad ones, are being assigned too generously to documents.",
        "min_tag_score": "Increase if you notice that the tagging results span too many domains.",
    }

    options = {}
    sidebar.header("Options")
    sidebar.subheader("Tag Tree")
    options["min_term_cluster_count"] = sidebar.slider(
        "Min document frequency", 0, 100, 4, help=tooltips["min_term_cluster_count"]
    )
    options["term_similarity_threshold"] = sidebar.slider(
        "Similarity threshold",
        0.6,
        1.0,
        0.95,
        help=tooltips["term_similarity_threshold"],
    )

    if include_tagging_options:
        sidebar.subheader("Document Tags")
        options["min_abstraction_similarity"] = sidebar.slider(
            "Min abstraction similarity",
            0.0,
            1.0,
            0.2,
            help=tooltips["min_abstraction_similarity"],
        )
        options["min_tag_score"] = sidebar.slider(
            "Min tag score", 0.0, 1.0, 0.15, help=tooltips["min_tag_score"]
        )

    return options


def main():

    # SIDEBAR
    dataset_name, view = create_sidebar(st.sidebar)

    # MAIN CONTAINER
    st.title("Data-driven topic hierarchies")

    if dataset_name == "None":
        # Do not show anything until a dataset is chosen
        st.text("Please select a dataset from the dropdown.")

    else:
        # Show results

        # Display relevant tuning parameters in the sidebar
        # Include additional options if using 'Document Search; view
        options = add_sidebar_options(
            st.sidebar, include_tagging_options=(view == "Document Search")
        )

        # Load HierarchicalTagger instance and extract hierarchical tree
        # Tree extraction is necessary for both app views.
        hierarchical_tagger = load_instance(dataset_name)
        hierarchical_tagger = copy.deepcopy(
            build_tag_tree(
                hierarchical_tagger,
                options["min_term_cluster_count"],
                options["term_similarity_threshold"],
            )
        )

        # Populate main container depending on View chosen
        if view == "Tag Tree":
            # Display hierarchical tag chart
            fig = build_sunburst(hierarchical_tagger)
            st.plotly_chart(fig, use_container_width=True)

        elif view == "Document Search":

            # Tag documents given extracted tag tree and parameters
            hierarchical_tagger = tag_documents(
                hierarchical_tagger,
                options["min_abstraction_similarity"],
                options["min_tag_score"],
            )

            # Two column display
            col1, col2 = st.columns(2)

            # Drop down to select tag
            nodes = [n.tag for n in hierarchical_tagger._core_nodes]
            tag = col1.selectbox("Choose a tag", nodes, index=0)
            node_idx = nodes.index(tag)

            # Drop down to select attribute
            attributes = hierarchical_tagger.document_attributes[list(hierarchical_tagger.document_attributes.keys())[0]].keys()
            attribute = col2.selectbox("Choose a document attribute to display", attributes, index=0)

            # Show information about most relevant documents
            top_docs = get_top_documents(hierarchical_tagger, node_idx, 20)
            for i, doc_id in enumerate(top_docs.keys()):
                if i % 2 == 0:
                    col1.subheader(doc_id)
                    col1.markdown("Topics:")
                    col1.json(top_docs[doc_id])
                    col1.markdown("Document attribute:")
                    col1.markdown(hierarchical_tagger.document_attributes[doc_id][attribute])
                else:
                    col2.subheader(doc_id)
                    col2.markdown("Topics:")
                    col2.json(top_docs[doc_id])
                    col2.markdown("Document attribute:")
                    col2.markdown(hierarchical_tagger.document_attributes[doc_id][attribute])



if __name__ == "__main__":
    main()
