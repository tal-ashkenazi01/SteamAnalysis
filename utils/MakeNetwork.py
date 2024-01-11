import networkx as nx
from itertools import combinations
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px


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
    user_network.update_layout(height=800, coloraxis={'colorscale': 'Aggrnyl'}, coloraxis_showscale=False,
                               legend={'itemsizing': 'constant', 'font': {'size': 10}})

    return user_network


def make_stacked_charts(input_data):
    # THE COLORS THAT ALL OF THE GRAPHS WILL USE
    games_flat_percent_share = [game for sublist in input_data for game in sublist]

    # THE GAME AND THEIR TOTAL COUNT
    game_counts = Counter(games_flat_percent_share)

    total_games = 0
    for number in game_counts.values():
        total_games = total_games + number

    games_to_show = []
    top_games = 0
    for key, val in game_counts.items():
        # IF THE GAME MAKES UP MORE THAN 10% OF THE TOTAL TOP LIBRARIES
        percent = val / total_games
        if percent >= .025:
            games_to_show.append([key, val, percent])
            top_games = top_games + val

    sorted_games_to_show = sorted(games_to_show, key=lambda x: x[1])
    number_remaining = total_games - top_games

    if number_remaining:
        percent_remaining = number_remaining / total_games
        sorted_games_to_show.insert(0, ["other", number_remaining, percent_remaining])

    # color_gradient = dict(zip([x for x in range(0, len(sorted_games_to_show))], px.colors.qualitative.G10))

    fig = go.Figure()

    cumulative_x = 0  # Initialize cumulative sum for x position
    for i, game in enumerate(sorted_games_to_show):
        x_value = game[1]  # Value for this segment
        percent_text = f"{game[2] * 100:.2f}%"
        fig.add_trace(go.Bar(
            x=[x_value], y=[1],
            name=game[0],
            hovertemplate='<span style="font-size: x-large;">&nbsp;&nbsp;<b>%{data.name}</b>&nbsp;&nbsp;</span><br>%{customdata}&nbsp;&nbsp;&nbsp;&nbsp;<extra></extra>',
            orientation='h',
            customdata=[percent_text]
        ))

        # Position annotation in the middle of the segment
        annotation_x = cumulative_x + x_value / 2
        cumulative_x += x_value  # Update cumulative sum

        if game[2] >= 0.035:
            # CHANGE HEIGHT HERE SO THAT THE ANNOTATIONS DON'T OVERLAP
            fig.add_annotation(dict(x=annotation_x, y=0.45,
                                    text=percent_text,
                                    font=dict(family='Arial', size=14),  # Adjust color for visibility: color='#333333'
                                    showarrow=False))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        hoverlabel_align='right',
        height=200,
        margin=dict(l=20, r=20, t=25, b=20)
    )

    fig.update_layout(title="Most Popular Games in Reviewers Libraries by Percent")
    return fig
