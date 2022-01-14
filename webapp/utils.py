import plotly.graph_objects as go


def build_sunburst(ht):
    ids, labels, values = zip(
        *[(n.identifier, str(n.tag), n.data.n_occurrences) for n in ht.tree.all_nodes()]
    )
    parents = [
        str(ht.tree.parent(id_).identifier) if ht.tree.parent(id_) is not None else ""
        for id_ in ids
    ]
    ids = [str(i) for i in ids]

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
    )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=700, width=700)

    return fig
