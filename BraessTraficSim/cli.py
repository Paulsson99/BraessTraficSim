from argparse import ArgumentParser

from BraessTraficSim.large_graph_automation import main as automation_main


def parse_args():
    parser = ArgumentParser("Run sim")
    parser.add_argument('-N', '--Network', type=int, action='extend', nargs='+', help="Specify network size", required=True)
    parser.add_argument('-n', '--nodes-to-connect', type=int, action='extend', nargs='+', help="Nodes to connect", required=True)
    parser.add_argument('-e', '--epochs', type=int, help="Epochs to run the model for", required=True)
    parser.add_argument('-s', '--show', action='store_true', help="Show the network configuration after each iteration")

    return parser.parse_args()


def main():
    args = parse_args()
    nodes_to_connect = [(args.nodes_to_connect[i], args.nodes_to_connect[i+1]) for i in range(0, len(args.nodes_to_connect), 2)]
    automation_main(network_size=args.Network, nodes_to_connect=nodes_to_connect, epochs=args.epochs, show_traffic_time=args.show)