APP_NAME = "FeaturePagerank"

import pyspark # pylint: disable=F0401,W0611
from operator import itemgetter
import argparse
import logging, sys, datetime, math, os, gc

logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_logging(output_path):
    log_filename = output_path + "/features-pagerank-" + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M") + ".log"
    log2file = logging.FileHandler(log_filename)
    log2file.setLevel(logging.DEBUG)
    log2file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(log2file)

def parse_hex(hex_id_string):
    """ Parse a hex id as it is written in the dataset files to a long, or to -1 if it evaluates to False. """
    return long(hex_id_string, 16) if hex_id_string else -1

def print_hex(hex_id):
    """ Print a hex id as it is written in the dataset files. """
    return hex(hex_id)[2:-1].upper().rjust(8, '0')

def add_all(nodes_to_value_rdd, nodes_rdd, empty_value):
    """ Returns an RDD where all nodes_rdd appear as keys,
    using nodes_to_value_rdd as a base and adding empty_value
    as value for those that did not appear there. """
    return (nodes_rdd
              .map(lambda n: (n, None))
              .leftOuterJoin(nodes_to_value_rdd)
              .mapValues(lambda (_, d): empty_value if d is None else d ))


def main(sc, basepath, output_dir,
        alpha=0.8, beta=0.1, delta_threshold=1E-9, max_number_of_iterations=100,
        n_top_to_save=1000, seed=123456789L, sample=None):
    output_path = output_dir + "/features-pagerank-" + datetime.datetime.now().strftime("%Y-%m-%d_%H.%M") + "/"
    os.mkdir(output_path)
    configure_logging(output_path)

    logger.info("Feature PageRank launched within Spark.\n\nSpark configuration:\n%s\n", sc._conf.toDebugString()) # pylint: disable=W0212
    logger.info("FeaturePageRank configuration:\n%s\n", '\n'.join(str(k) + "=" + str(v) for (k,v) in locals().copy().items()))


    logger.info("Reading input files and computing ids of all the considered nodes...")
    hex_links = (sc.textFile(basepath + "PaperReferences.txt")
     .map(lambda line : tuple(parse_hex(t) for t in line.split("\t"))))

    # Hex ids of all the nodes in graph
    hex_in_graph = hex_links.flatMap(lambda x: x).distinct()

    def _parse_features(line):
        s = line.split("\t", 3)
        return tuple(parse_hex(t) for t in (s[0], s[2]))

    hexnode2hexfeature = (sc.textFile(basepath + "PaperAuthorAffiliations.txt")
     .map(_parse_features).filter(lambda x: -1 not in x))
    hex_nodes_with_features = hexnode2hexfeature.keys().distinct()

    # We will consider only hex ids appearing here
    considered_hex_ids = hex_in_graph.intersection(hex_nodes_with_features)
    if sample is not None and sample < 1.:
        considered_hex_ids = considered_hex_ids.sample(False, sample, seed)
    num_nodes = considered_hex_ids.count()

    # Python dict from an original hex id to our node id
    hex2id = dict((v,k) for (k,v) in enumerate(considered_hex_ids.toLocalIterator()))

    assert num_nodes == len(hex2id)
    nodes = sc.parallelize(xrange(num_nodes))
    logger.info("%d considered nodes.", num_nodes)

    logger.info("Computing the association between features and nodes...")
    # Let us consider only the relevant node -> feature association
    node2hexfeat_arcs = hexnode2hexfeature.map(
        lambda (hn, hf) : ( hex2id.get(hn,-1), hf )
    ).filter(lambda (n, _): n != -1)
    node2hexfeats = add_all(node2hexfeat_arcs.groupByKey(), nodes, [])
    hexfeat2nodes = node2hexfeat_arcs.map(reversed).groupByKey()
    hex_features = hexfeat2nodes.keys().distinct()

    feat_per_node_stats = node2hexfeats.values().map(len).stats()
    logger.info("%d features.", hex_features.count())
    logger.info("Number of features per node statistics: %s", feat_per_node_stats)


    logger.info("Computing the graph and the transposed graph...")
    links = hex_links.map(
        lambda x : tuple(hex2id.get(t,-1) for t in x)
    ).filter(lambda x: -1 not in x)

    # The graph: an RDD with nodes as keys and list of successors as values
    graph = add_all(links.groupByKey(), nodes, [])
    # The transposed graph: an RDD with nodes as keys and list of predecessors as values
    transposed_graph = add_all(links.map(reversed).groupByKey(), nodes, [])

    # RDD from nodes to their outdegree
    outdegrees = graph.mapValues(len)
    logger.info("Outdegrees statistics: %s", outdegrees.values().stats())


    # The (initial) rank: an RDD from node to its rank (initially uniform)
    uniform = 1. / num_nodes
    rank = nodes.map(lambda node: (node, uniform) )

    features_rank = hex_features.map(lambda f: (f, 0))
    hexfeat2degree = hexfeat2nodes.mapValues(len)
    num_nodes_and_features = num_nodes + hex_features.count()


    logger.info("Rank initialized. Starting pagerank iteration...")

    # Clear up some variables to free memory
    hex2id = None
    links.unpersist()
    del links
    hex_links.unpersist()
    del hex_links
    hex_in_graph.unpersist()
    del hex_in_graph
    hexnode2hexfeature.unpersist()
    del hexnode2hexfeature
    hex_features.unpersist()
    del hex_features
    hex_nodes_with_features.unpersist()
    del hex_nodes_with_features
    considered_hex_ids.unpersist()
    del considered_hex_ids
    gc.collect()

    for iteration in range(max_number_of_iterations):

        logger.info("Iteration %d. Computing each node contribution and broadcasting it...", iteration)
        # RDD with the rank contribution of each node to its neighbors
        # (treating dangling nodes as neighbors of everybody)
        contribution_rdd = rank.join(outdegrees).mapValues(
            lambda (r, outdeg): r / (outdeg if outdeg > 0 else num_nodes)
        )
        # We need it broadcasted
        bc_contributions = sc.broadcast(tuple(contribution_rdd.sortByKey().values().collect()))


        rdd_feat_contributions = features_rank.join(hexfeat2degree).mapValues(lambda (r, deg): r / deg)
        bc_feat_contributions = sc.broadcast(rdd_feat_contributions.collectAsMap())


        # Compute the total contributions of the dangling nodes

        logger.info("Iteration %d. Computing dangling nodes total contribution...", iteration)
        dangling_contribute = (contribution_rdd             # Take all contributions...
                .leftOuterJoin(outdegrees)
                .filter(lambda (_, (__, outd)) : outd == 0) # ...of dangling nodes...
                .values().map(itemgetter(0)).sum()          # ...and sum it.
        )

        stability = (1. - alpha) / num_nodes_and_features
        common_node_rank = alpha * dangling_contribute + stability


        logger.info("Iteration %d. Computing new rank...", iteration)
        # Map function to compute the new rank from the transposed_graph
        def compute_rank((preds, feats)):
            return ( common_node_rank +
                 alpha * beta       * sum(bc_contributions.value[j] for j in preds)
             +   alpha * (1 - beta) * sum(bc_feat_contributions.value[k] for k in feats )
            )

        # Compute new rank
        new_rank = transposed_graph.join(node2hexfeats).mapValues(compute_rank)

        # Update feature ranks
        features_rank = hexfeat2nodes.mapValues(
            lambda nodes: stability
                + alpha * sum(bc_contributions.value[j] for j in nodes)
        )


        logger.info("Iteration %d. Computing delta...", iteration)
        # Compute difference among nodes, and switch new and old ranks
        differences = (rank.join(new_rank).values()
                      .map(lambda (old, new): abs(new - old))
                      )
        difference = differences.sum()
        rank = new_rank
        del new_rank
        logger.info("Iteration %d. Delta = %f", iteration, difference)
        logger.info("Iteration %d. Node rank statistics: %s", iteration, rank.values().stats())
        logger.info("Iteration %d. Feat rank statistics: %s", iteration, features_rank.values().stats())
        logger.info("Iteration %d. Difference statistics: %s", iteration, differences.stats())

        bc_contributions.unpersist()
        bc_feat_contributions.unpersist()

        logger .info("Iteration %d. Saving node rank...", iteration)
        rank         .saveAsTextFile("%s/node-rank-%04d.txt" % (output_path, iteration))
        logger .info("Iteration %d. Saving feat rank...", iteration)
        features_rank.saveAsTextFile("%s/feat-rank-%04d.txt" % (output_path, iteration))
        logger.info("Iteration %d. Output saved in %s.", iteration, output_path)

        if difference < delta_threshold:
            logger.info("Iteration %d. Delta %10f smaller than %10f: stopping.", iteration, difference, delta_threshold)
            break

    logger.info("Saving top %d feature names...", n_top_to_save)

    def _parse_names(line):
        s = line.split('\t', 1)
        return (parse_hex(s[0]), s[1].title())

    hexfeature2names = sc.textFile(basepath + "Affiliations.txt").map(_parse_names)

    with open("%s/top-%d.tsv" % (output_path, n_top_to_save), "w") as output_file:
        for pos, (h, (r, n)) in enumerate(features_rank.join(hexfeature2names).top(n_top_to_save, key=lambda x: x[1][0])):
            output_file.write("%05d\t%f\t%s\t%s\n" % (pos, math.log(r), print_hex(h), n.encode('UTF-8')))

    logger.info("Done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('basepath', help='Basepath of the data set files.')
    parser.add_argument('output', help='Shared directory where to save output.')
    parser.add_argument('master', help='Spark master.')
    parser.add_argument('--alpha', default=.8, type=float,
         help='Dumping factor for PageRank [default: .8]')
    parser.add_argument('--beta', default=0.05, type=float,
         help='Amount of feature bipartite graph inside the union graph. [default: .1]')
    parser.add_argument('--delta', default=1E-12, type=float,
         help='Stopping threshold for delta between rank vectors. [default: 1E-9]')
    parser.add_argument('--sample', type=float,
         help='If provided with a float, the considered nodes will be sampled with that fraction.')
    parser.add_argument('--seed', type=long, default=123456789L,
         help='This seed will be used for random extraction (in sampling) [default: 123456789].')
    parser.add_argument('--max', default=200, type=int,
         help='Maximum number of iteration if the stopping threshold is not reached. [default: 200]')
    parser.add_argument('--top', default=5000, type=int,
         help='Number of top features names to save as a TSV file. [default: 1000]')

    args = parser.parse_args()


    conf = pyspark.conf.SparkConf().setAll([
        ('spark.akka.frameSize',       '1024'),
        ('spark.default.parallelism',  '24'),
        ('spark.executor.cores',       '4'),
        ('spark.executor.instances',   '16'),
        ('spark.driver.memory',        '55G'),
        ('spark.rpc.askTimeout',       '600'),
        ('spark.executor.memory',      '60G'),
        ('spark.submit.deployMode',    'client'),
        ('spark.driver.maxResultSize', '25g'),
        ('spark.app.name',             APP_NAME),
        ('spark.master',               args.master)
    ])

    main(
        pyspark.SparkContext(conf=conf),
        args.basepath, args.output,
        alpha=args.alpha, beta=args.beta, delta_threshold=args.delta,
        max_number_of_iterations=args.max, n_top_to_save=args.top,
        sample=args.sample, seed=args.seed)
