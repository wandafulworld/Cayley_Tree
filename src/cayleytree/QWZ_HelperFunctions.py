"""
Provides functions that help construct the adjacency matrix of the Qi-Wu-Zhang (QWZ) model on a Cayley tree.

Because the edges connecting node a to node b depend on the relative locations of node a and b, we need each
node to carry its relative location as an attribute.

"""

import networkx as nx
from typing import List, Dict

def qwz_connection_builder(G,parent_nodes,next_nodes: List[int],relative_location: str,t_sigma=1,scale=0.5):
    # next nodes contains the next 6 nodes in a list in an ordered fashion which we connect to the parent
    # implicitly assumes that ordering is with odd first and even second of parent nodes and next_nodes
    # assumes that nodes are already there and only adds edges
    if relative_location == "south":
        # South (t_y dagger)
        locator = dict((el, 'south') for el in next_nodes[0:2])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[0],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[1], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[1], weight=-scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[0], weight=scale*t_sigma)

        # other Direction
        G.add_edge(next_nodes[0],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[1], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[0], parent_nodes[1], weight=scale * t_sigma)
        G.add_edge(next_nodes[1], parent_nodes[0], weight=-scale * t_sigma)

        # East (t_x)
        locator = dict((el, 'east') for el in next_nodes[2:4])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[2],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[3], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[3], weight=1j*scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[2], weight=1j*scale*t_sigma)

        G.add_edge(next_nodes[2],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[3], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[2], parent_nodes[1], weight=-1j*scale * t_sigma)
        G.add_edge(next_nodes[3], parent_nodes[0], weight=-1j*scale * t_sigma)

        #west (t_x dagger) we skip north because relative location south
        locator = dict((el, 'west') for el in next_nodes[4:])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[4],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[5], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[5], weight=-1j*scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[4], weight=-1j*scale*t_sigma)

        G.add_edge(next_nodes[4],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[5], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[4], parent_nodes[1], weight=1j*scale * t_sigma)
        G.add_edge(next_nodes[5], parent_nodes[0], weight=1j*scale * t_sigma)

    elif relative_location == "east":
        # South (t_y dagger)
        locator = dict((el, 'south') for el in next_nodes[0:2])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[0], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[1], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[1], weight=-scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[0], weight=scale * t_sigma)

        # other Direction
        G.add_edge(next_nodes[0],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[1], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[0], parent_nodes[1], weight=scale * t_sigma)
        G.add_edge(next_nodes[1], parent_nodes[0], weight=-scale * t_sigma)

        # east (t_x)
        locator = dict((el, 'east') for el in next_nodes[2:4])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[2], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[3], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[3], weight=1j * scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[2], weight=1j * scale * t_sigma)

        G.add_edge(next_nodes[2],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[3], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[2], parent_nodes[1], weight=-1j*scale * t_sigma)
        G.add_edge(next_nodes[3], parent_nodes[0], weight=-1j*scale * t_sigma)

        # north (t_y)
        locator = dict((el, 'north') for el in next_nodes[4:])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[4], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[5], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[5], weight=scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[4], weight= -scale * t_sigma)

        G.add_edge(next_nodes[4],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[5], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[4], parent_nodes[1], weight=-scale * t_sigma)
        G.add_edge(next_nodes[5], parent_nodes[0], weight=scale * t_sigma)

    elif relative_location == 'north':
        # east (t_x)
        locator = dict((el, 'east') for el in next_nodes[0:2])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[0], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[1], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[1], weight=1j * scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[0], weight=1j * scale * t_sigma)

        G.add_edge(next_nodes[0],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[1], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[0], parent_nodes[1], weight=-1j*scale * t_sigma)
        G.add_edge(next_nodes[1], parent_nodes[0], weight=-1j*scale * t_sigma)

        # north (t_y)
        locator = dict((el, 'north') for el in next_nodes[2:4])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[2], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[3], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[3], weight=scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[2], weight=-scale * t_sigma)

        G.add_edge(next_nodes[2],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[3], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[2], parent_nodes[1], weight=-scale * t_sigma)
        G.add_edge(next_nodes[3], parent_nodes[0], weight=scale * t_sigma)

        #west (t_x dagger)
        locator = dict((el, 'west') for el in next_nodes[4:])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[4],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[5], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[5], weight=-1j*scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[4], weight=-1j*scale*t_sigma)

        G.add_edge(next_nodes[4],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[5], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[4], parent_nodes[1], weight=1j*scale * t_sigma)
        G.add_edge(next_nodes[5], parent_nodes[0], weight=1j*scale * t_sigma)

    elif relative_location == 'west':
        # South (t_y dagger)
        locator = dict((el, 'south') for el in next_nodes[0:2])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[0],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[1], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[1], weight=-scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[0], weight=scale*t_sigma)

        # other Direction
        G.add_edge(next_nodes[0], parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[1], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[0], parent_nodes[1], weight=scale * t_sigma)
        G.add_edge(next_nodes[1], parent_nodes[0], weight=-scale * t_sigma)

        # north (t_y)
        locator = dict((el, 'north') for el in next_nodes[2:4])
        nx.set_node_attributes(G, locator,name='location')
        G.add_edge(parent_nodes[0], next_nodes[2], weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[3], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[3], weight=scale * t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[2], weight=-scale * t_sigma)

        G.add_edge(next_nodes[2],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[3], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[2], parent_nodes[1], weight=-scale * t_sigma)
        G.add_edge(next_nodes[3], parent_nodes[0], weight=scale * t_sigma)

        # west (t_x dagger)
        locator = dict((el, 'west') for el in next_nodes[4:])
        nx.set_node_attributes(G,locator,name='location')
        G.add_edge(parent_nodes[0],next_nodes[4],weight=-scale)
        G.add_edge(parent_nodes[1], next_nodes[5], weight=scale)
        G.add_edge(parent_nodes[0], next_nodes[5], weight=-1j*scale*t_sigma)
        G.add_edge(parent_nodes[1], next_nodes[4], weight=-1j*scale*t_sigma)

        G.add_edge(next_nodes[4],parent_nodes[0], weight=-scale)
        G.add_edge(next_nodes[5], parent_nodes[1], weight=scale)
        G.add_edge(next_nodes[4], parent_nodes[1], weight=1j*scale * t_sigma)
        G.add_edge(next_nodes[5], parent_nodes[0], weight=1j*scale * t_sigma)

    else:
        print('Error: Location marker is missing')




def qwz_inner_ring(G,center_nodes,next_nodes,t_sigma,scale=0.5):
    """
    For a QWZ Cayley tree, this function initializes the first 4 (x2 = 8) nodes around the center
    As these do not work with a relative location reference. The function gives the nodes their relative location
    and adds hopping in both directions
    :param G: The empty DiGraph object representing the QWZ graph
    :param center_nodes: the two central nodes
    :param next_nodes: the 8 nodes around the center
    :param t_sigma: hopping parameter for the mixing
    :param scale: Set to 1/2 for the QWZ, scales all matrices
    :return: None, things are directly added to G
    """
    # South
    locator = dict((el, 'south') for el in next_nodes[:2])
    nx.set_node_attributes(G, locator,name='location')
    G.add_edge(center_nodes[0], next_nodes[0], weight=-scale)
    G.add_edge(center_nodes[1], next_nodes[1], weight=scale)
    G.add_edge(center_nodes[0], next_nodes[1], weight=-scale * t_sigma)
    G.add_edge(center_nodes[1], next_nodes[0], weight=scale * t_sigma)

    # other Direction
    G.add_edge(next_nodes[0], center_nodes[0], weight=-scale)
    G.add_edge(next_nodes[1], center_nodes[1], weight=scale)
    G.add_edge(next_nodes[0], center_nodes[1], weight=scale * t_sigma)
    G.add_edge(next_nodes[1], center_nodes[0], weight=-scale * t_sigma)

    # East
    locator = dict((el, 'east') for el in next_nodes[2:4])
    nx.set_node_attributes(G, locator,name='location')
    G.add_edge(center_nodes[0], next_nodes[2], weight=-scale)
    G.add_edge(center_nodes[1], next_nodes[3], weight=scale)
    G.add_edge(center_nodes[0], next_nodes[3], weight=1j * scale * t_sigma)
    G.add_edge(center_nodes[1], next_nodes[2], weight=1j * scale * t_sigma)

    G.add_edge(next_nodes[2], center_nodes[0], weight=-scale)
    G.add_edge(next_nodes[3], center_nodes[1], weight=scale)
    G.add_edge(next_nodes[2], center_nodes[1], weight=-1j * scale * t_sigma)
    G.add_edge(next_nodes[3], center_nodes[0], weight=-1j * scale * t_sigma)

    # North
    locator = dict((el, 'north') for el in next_nodes[4:6])
    nx.set_node_attributes(G, locator,name='location')
    G.add_edge(center_nodes[0], next_nodes[4], weight=-scale)
    G.add_edge(center_nodes[1], next_nodes[5], weight=scale)
    G.add_edge(center_nodes[0], next_nodes[5], weight=scale * t_sigma)
    G.add_edge(center_nodes[1], next_nodes[4], weight=-scale * t_sigma)

    G.add_edge(next_nodes[4], center_nodes[0], weight=-scale)
    G.add_edge(next_nodes[5], center_nodes[1], weight=scale)
    G.add_edge(next_nodes[4], center_nodes[1], weight=-scale * t_sigma)
    G.add_edge(next_nodes[5], center_nodes[0], weight=scale * t_sigma)

    # West
    locator = dict((el, 'west') for el in next_nodes[6:])
    nx.set_node_attributes(G, locator,name='location')
    G.add_edge(center_nodes[0], next_nodes[6], weight=-scale)
    G.add_edge(center_nodes[1], next_nodes[7], weight=scale)
    G.add_edge(center_nodes[0], next_nodes[7], weight=-1j * scale * t_sigma)
    G.add_edge(center_nodes[1], next_nodes[6], weight=-1j * scale * t_sigma)

    G.add_edge(next_nodes[6], center_nodes[0], weight=-scale)
    G.add_edge(next_nodes[7], center_nodes[1], weight=scale)
    G.add_edge(next_nodes[6], center_nodes[1], weight=1j * scale * t_sigma)
    G.add_edge(next_nodes[7], center_nodes[0], weight=1j * scale * t_sigma)



