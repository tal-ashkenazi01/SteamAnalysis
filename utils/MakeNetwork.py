import networkx as nx
from itertools import combinations
import plotly.graph_objects as go


def make_network(input_list):
    G = nx.Graph()

    # Add edges
    for games in input_list:
        for game1, game2 in combinations(games, 2):
            G.add_edge(game1, game2)

    pos = nx.spring_layout(G)

    # Extract node and edge information from the graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None is added to create a discontinuity between edges
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Create the Plotly graph
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False)

    # Add node labels with node names
    node_adjacencies = []
    node_name = []
    node_text = []

    # Iterate over each node and its adjacency list
    for node, adjacencies in G.adjacency():
        num_connections = len(adjacencies)
        node_adjacencies.append(num_connections)
        # Update this line to include node name/identifier
        node_name.append(node)
        node_text.append(f'<b>Game: {node}</b><br># of connections: {num_connections}')

    # Create the figure
    user_network = go.Figure(data=[edge_trace],
                             layout=go.Layout(
                                 showlegend=True,
                                 hovermode='closest',
                                 margin=dict(b=20, l=5, r=5, t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                             )

    for i in range(0, len(node_x)):
        user_network.add_trace(
            go.Scatter(x=[node_x[i]], y=[node_y[i]], name=node_name[i], text=[node_text[i]], mode='markers',
                       hoverinfo='text',
                       marker=dict(size=15, line_width=1, showscale=False,
                                   coloraxis="coloraxis", color=[node_adjacencies[i]])))
    user_network.update_layout(height=800, coloraxis={'colorscale': 'viridis'}, coloraxis_showscale=False,
                               legend={'itemsizing': 'constant'})

    return user_network
