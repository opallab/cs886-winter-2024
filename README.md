# CS 886: Graph Neural Networks

## Logistics
+ **Instructor:** [Kimon Fountoulakis](https://opallab.ca/team/)
+ **Seminar Time:** 10:00-11:20 AM, Monday and Wednesday
+ **Office hours:** 1:00-2:00 PM, Tuesday

## Overview

Learning from multi-modal datasets is currently one of the most prominent topics in artificial intelligence. The reason behind this trend is that many applications, such as recommendation systems and fraud detection, require the combination of different types of data. In addition, it is often the case that data exhibit relations which need to be captured for downstream applications. In this proposal we are interested in multi-modal data which combine a graph, i.e., a set of nodes and edges, with attributes for each node and/or edge. The attributes of the nodes/edges capture information about the nodes/edges themselves, while the edges among the nodes capture relations among the nodes. Capturing relations is particularly helpful for applications where we are trying to make predictions for nodes given neighborhood data.

One of the most prominent and principled ways of handling such multi-modal data for downstream tasks such as node classification is graph neural networks. Graph neural network models can mix hand-crafted or automatically learned attributes about the nodes while taking into account relational information among the nodes. Therefore, the output vector representation of the graph neural network contains global and local information for the nodes. This contrasts with neural networks that only learn from the attributes of entities. 

This seminar will cover seminal work in the space of graph neural networks. For example, spectral and spatial convolutional graph neural networks, graph attention networks, invariant and equivariant graph neural networks, general message passing graph neural networks. We will focus on both practical and theoretical aspects of graph neural networks. Practical aspects include, scalability and performance on real data. Examples of theoretical questions include: what does convolution do to the input data? Does convolution improve generalization compared to not using a graph? How do multiple convolutions change the data and how do they affect generalization?

The seminar is based on weekly paper readings and student presentations, discussions, and
a term project. 

## (Tentative) Schedule
The schedule below is subject to change:
| Week | Date | Topic | Readings | Slides |
|:-----|:-----|:------|:------------|:-----|
| 1 | 1/8 | Introduction, problems and applications (Kimon lecturing) | [Geometric Deep Learning](https://arxiv.org/abs/2104.13478) (Chapter 1) <br/> [Geometric foundations of Deep Learning](https://towardsdatascience.com/towards-geometric-deep-learning-iv-chemical-precursors-of-gnns-11273d74125) <br/>  [Towards Geometric Deep Learning I: On the Shoulders of Giants](https://towardsdatascience.com/towards-geometric-deep-learning-i-on-the-shoulders-of-giants-726c205860f5) <br/> [Towards Geometric Deep Learning II: The Perceptron Affair](https://towardsdatascience.com/towards-geometric-deep-learning-ii-the-perceptron-affair-fafa61b5c40a) <br/> [Towards Geometric Deep Learning III: First Geometric Architectures](https://towardsdatascience.com/towards-geometric-deep-learning-iii-first-geometric-architectures-d1578f4ade1f) <br/> [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) <br/> [Intro to graph neural networks (ML Tech Talks)](https://www.youtube.com/watch?v=8owQBFAHw7E) <br/> [Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_1.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_1.key)  |
| 1 | 1/10 | Spatial graph convolution and its theoretical performance on simple random data, Part 1 (Kimon lecturing) | This lecture is based on 1) [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) <br/> 2) [Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization](https://proceedings.mlr.press/v139/baranwal21a.html), [Video](https://zoom.us/rec/play/X1FbBJiP1bLTixjmU7wBw233sutk939XulBkrY0Szes9KSNh_cFovdohKoZ-KXFaCZJ5G5yg4m5nKZol.47Ol60UmzMVLZec8?startTime=1624287370000&_x_zm_rtaid=2ArAs6KUSwiFwnA5V61cmQ.1624792221031.e0fb3030146eeed7cee824bfc92e70b5&_x_zm_rhtaid=77) (time 1:03:34), [Code for reproducing the experiments](https://github.com/opallab/Graph-Convolution-for-Semi-Supervised-Classification-Improved-Linear-Separability-and-OoD-Gen.) <br/> Related material are: [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697) <br/> [PyTorch code for GCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv), [Example code](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_2.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_2.key) |
| 2 | 1/15 | Spatial graph convolution and its theoretical performance on simple random data, Part 2 (Kimon lecturing) | This lecture is based on [Effects of Graph Convolutions in Multi-layer Networks](https://openreview.net/pdf?id=P-73JPgRs0R), [Code for reproducing the experiments](https://github.com/opallab/Effects-of-Graph-Convs-in-Deep-Nets) <br/> Related material are: [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697) <br/> [PyTorch code for GCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv), [Example code](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_3.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_3.key) |
| 2 | 1/17 | Graph Attention Network and Graph Attention Retrospective (Kimon lecturing) | This lecture is based on 1) [Graph Attention Networks](https://arxiv.org/abs/1710.10903), [PyTorch Code](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv) <br/> and 2) [Graph Attention Retrospective](https://jmlr.org/papers/v24/22-125.html), [Code for reproducing the experiments](https://github.com/opallab/Graph-Attention-Retrospective/), [Video lecture](https://youtu.be/duWVNO8_sDM) <br/> A related paper is: [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/lecture_4.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_4.key) | 
| 3 | 1/22 | Optimality of Message Passing (Kimon lecturing)| This lecture is based on [Optimality of Message-Passing Architectures for Sparse Graphs](https://openreview.net/forum?id=d1knqWjmNt) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_5.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_5.key) | 
| 3 | 1/24 | Message Passing, Symmetries and Reasoning (Kimon lecturing)| This lecture is based on [What Can Neural Networks Reason About?](https://arxiv.org/abs/1905.13211) and [Geometric Deep Learning](https://arxiv.org/abs/2104.13478)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_6.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_6.key) | 
| 4 | 1/29 | The first graph neural network model and a popular spectral graph convolution model | [The Graph Neural Network Model](https://ieeexplore.ieee.org/document/4700287) and [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)| [PDF 1](https://github.com/opallab/cs886-winter-2024/blob/main/The_Graph_Neural_Network_Model.pdf), [PDF 2](https://github.com/opallab/cs886-winter-2024/blob/main/cnns_on_graphs.pdf) |
| 4 | 1/31 | Introduction to the expressive power of graph neural networks| [Expressive power of graph neural networks and the Weisfeiler-Lehman test](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49) and [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826). A free alternative to the blogpost: [The Expressive Power of Graph Neural Networks (Chapter 5.3)](https://graph-neural-networks.github.io/static/file/chapter5.pdf) |  [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_expressivity.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_expressivity.key) |
| 5 | 2/5 | Expressive power, Part 2 | [The Expressive Power of Graph Neural Networks (Chapter 5.4, up to 5.4.2.3), included)](https://graph-neural-networks.github.io/static/file/chapter5.pdf) and [What graph neural networks cannot learn: depth vs width](https://openreview.net/pdf?id=B1l2bp4YwS)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/TheExpressivePowerofGraphNeuralNetworks.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/WhatGraphNeuralNetworksCannotLearn.pdf) | 
| 5 | 2/7 | Higher-order graph neural networks | [k-hop Graph Neural Networks](https://arxiv.org/abs/1907.06051) and [Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/K-hopGNNs.pdf), [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/MixHop.pptx) |
| 6 | 2/12 | Higher-order graph neural networks and their expressive power | [Multi-Hop Attention Graph Neural Network](https://arxiv.org/abs/2009.14332) and [Provably Powerful Graph Networks](https://arxiv.org/abs/1905.11136) and [How Powerful are K-hop Message Passing Graph Neural Networks](https://arxiv.org/pdf/2205.13328.pdf)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_provably_powerful_gnns.key), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/HowPowerfulareK-hopMessagePassingGraph%20NeuralNetworks.pdf) |
| 6 | 2/14 | Invariant and Equivariant Graph Neural Networks, Part 1 | [An Introduction To Invariant Graph Networks, Part 1](http://irregulardeep.org/An-introduction-to-Invariant-Graph-Networks-(1-2)/) and [An Introduction To Invariant Graph Networks, Part 2](https://irregulardeep.org/How-expressive-are-Invariant-Graph-Networks-(2-2)/) and the corresponding paper [Invariant and Equivariant Graph Networks](https://arxiv.org/pdf/1812.09902.pdf) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/invariant_and_equivariant_gnns.pdf) |
| 7 | 2/19 | Reading Week |  | | |
| 7 | 2/21 | Reading Week |  | | |
| 8 | 2/26 | Invariant and Equivariant Graph Neural Networks, Part 2 | [Geometric Deep Learning (Chapter 5.5)](https://arxiv.org/pdf/2104.13478.pdf) and [Building powerful and equivariant graph neural networks with structural message-passing](https://proceedings.neurips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf) and [Universal Invariant and Equivariant Graph Neural Networks](https://papers.nips.cc/paper/2019/file/ea9268cb43f55d1d12380fb6ea5bf572-Paper.pdf)  | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/universal_invariant_and_equivariant_ggns.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/building_powerful_and_equivariant_gnns.pdf) |
| 8 | 2/28 | GNNs for heterogeneous graphs | [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) and [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/abs/2002.01680) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/rgcn.pdf), [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/magnn.pptx) |
| 9 | 3/4 | Oversmoothing Part 1 | [Over-smoothing issue in graph neural network](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472) and [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751)| [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/oversmoothing_issue.pptx), [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/DeepGCNs.pptx) |
| 9 | 3/6 | Oversmoothing Part 2 | [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) and [Not too little, not too much: a theoretical analysis of graph (over)smoothing](https://arxiv.org/abs/2205.12156)| [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/simple_and_deep_gcns.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/not_too_little_not_too_much_theory.pdf) |
| 10 | 3/11 | Scalable GNNs Part 1 | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) and [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) |  [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/ClusterGCN_Presentation.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_SIGN.key) |
| 10 | 3/13 | Scalable GNNs Part 2 | [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) and [Training Graph Neural Networks with 1000 Layers](https://arxiv.org/pdf/2106.07476.pdf) |[Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_SAINT.key), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/one_thousand_layers.pdf)  | 
| 11 | 3/18 | Self-supervised learning in graphs | [Graph Self-Supervised Learning: A Survey](https://arxiv.org/abs/2103.00111) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/GraphSSL1.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/graph_SSL_part_2.pdf) |
| 11 | 3/20 | GNNs for combinatorial optimization Part 1 | [Erdos Goes Neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs](https://proceedings.neurips.cc/paper/2020/file/49f85a9ed090b20c8bed85a5923c669f-Paper.pdf) and [Simulation of Graph Algorithms with Looped Transformers](https://arxiv.org/abs/2402.01107) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Erdos_goes_neural.pdf), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/Lecture_Simulation.key) |
| 12 | 3/25 | GNNs for combinatorial optimization Part 2 | [Attention, Learn to Solve Routing Problems!](https://arxiv.org/pdf/1803.08475.pdf) and [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/file/d14c2267d848abeb81fd590f371d39bd-Paper.pdf) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/attention_learn_to_solve_co.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/exact_combinatorial.pdf) |
| 12 | 3/27 | Link prediction | [Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) and [Line Graph Neural Networks for Link Prediction](https://arxiv.org/pdf/2010.10046.pdf) | [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/Link_Prediction_Based_on_Graph_Neural_Networks.pptx), [Keynote](https://github.com/opallab/cs886-winter-2024/blob/main/LineGNNLinkPrediction.key) |
| 13 | 4/1 | Algorithmic Reasoning | [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761) and [Pointer Graph Networks](https://arxiv.org/abs/2006.06380) and [A Generalist Neural Algorithmic Learner](https://arxiv.org/abs/2209.11142)| [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/Neural_Algorithm_Reasoning.pptx), [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/Pointer_Graph_Networks.pptx), [PPT](https://github.com/opallab/cs886-winter-2024/blob/main/A_Generalist_Neural_Algorithmic_Learner.pptx) |
| 13 | 4/3 | Generative GNNs | [Chapter 11: Graph Neural Networks: Graph Generation](https://graph-neural-networks.github.io/gnnbook_Chapter11.html) | [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Generative_GNNs_Part_I.pdf), [PDF](https://github.com/opallab/cs886-winter-2024/blob/main/Generative_GNNs_Part_II.pdf) |

## Readings

+ [Geometric Deep Learning](https://geometricdeeplearning.com), Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković, 2021
+ [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697), Stefanie Jegelka, 2022
+ [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/), William L. Hamilton, 2020
+ [Graph Neural Networks](https://graph-neural-networks.github.io), Lingfei Wu, Peng Cui, Jian Pei, Liang Zhao, (2022)

## Other courses online related to machine learning on graphs

+ [Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/), Jure Leskovec, Stanford
+ [Graph Representation Learning](https://cs.mcgill.ca/~wlh/comp766/), William L. Hamilton, McGill
+ [Introduction to Graph Neural Networks](https://www.youtube.com/watch?v=Iiv9R6BjxHM), Xavier Bresson, Nanyang Techinical University and NYU
+ [Recent Developments in Graph Network Architectures](https://www.youtube.com/watch?v=M60huxIvKbE), Xavier Bresson, Nanyang Techinical University
+ [Benchmarking GNNs](https://www.youtube.com/watch?v=tuChBSo8_eg), Xavier Bresson, Nanyang Techinical University
+ [Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc), Petar Veličković, DeepMind
+ [Geometric Deep Learning Course](https://geometricdeeplearning.com/lectures/)
+ [Machine Learning for the Working Mathematician: Geometric Deep Learning](https://www.youtube.com/watch?v=7pRIjJ_u2_c), Geordie Williamson, The University of Syndney
+ [Advanced lectures on community detection](https://indico.ictp.it/event/9797/other-view?view=ictptimetable), Laurent Massoulie, INRIA Paris

## Online reading seminars

+ [LoGaG: Learning on Graphs and Geometry Reading Group](https://hannes-stark.com/logag-reading-group)

## Code

+ [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
+ [Deep Graph Library](https://www.dgl.ai)

## Competitions

+ [Open Graph Benchmark](https://ogb.stanford.edu/docs/leader_overview/)

## Datasets

+ [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
+ [Open Graph Benchmark](https://ogb.stanford.edu)
+ [HyperGraphs](https://www.cs.cornell.edu/~arb/data/)
+ [TUDatasets](https://chrsmrrs.github.io/datasets/)
+ [Non Homophily Benchmarks](https://github.com/CUAI/Non-Homophily-Benchmarks)
+ [Graph Learning Benchmarks](https://graph-learning-benchmarks.github.io/glb2022)
+ [Hetionet](https://het.io)
+ [Heterogeneous graph benchmarks](https://www.biendata.xyz/hgb/)
+ [Long Range Graph Benchmark](https://towardsdatascience.com/lrgb-long-range-graph-benchmark-909a6818f02c)
+ [IGB-Datasets](https://github.com/IllinoisGraphBenchmark/IGB-Datasets)


## Workload Breakdown
+ Class Participation: 15%
+ Paper Reviews: 20%
+ Presentation: 25%
+ Project: 40%

## Paper Reviews
After the first 6 seminars, I will require one review per week for one of the papers that have be assigned to that week. If there are more than one papers assigned, you can pick any of 
the assigned papers. You are allowed to skip 2 reviews throughout the term.  The reviews will be 2 pages long (if you need more space take another 0.25 page but try not to). 
You have to finish your review with one question. The reviews are due at 11:59pm Friday. Submit your review on Crowdmark. Each review has the same weight.

You are expected to answer the following 5 questions and finish your reviews with a
question that can start a discussion in class:

+ What is the problem?
+ Why is it important?
+ Why don't previous methods work on that problem?
+ What is the solution to the problem the authors propose?
+ What interesting research questions does the paper raise?

There is no fixed format for the reviews but I recommend: Single column, 1.5 space, 12 pt, in Latex.
Ultimately, the main thing I am looking for is a demonstration of serious critical reading of the paper.

## Project Deliverables
There is one main deliverable of your project, a 6-page paper and (if relevant) the source code of your project 
with instructions to run your code. I provide a few options for the projects below:

+ Option A (Empirical evaluation):
Pick a problem that interests you.
Implement and experiment with several graph neural network methods to tackle this problem.
+ Option B (Method design):
Identify a problem for which there are no satisfying approaches.
Develop a new graph neural network architecture to tackle this problem.
Analyze theoretically and/or empirically the performance of your technique.
+ Option C (Theoretical analysis):
Identify a problem or a graph neural network architecture for which theoretical performance (e.g., complexity, performance on random data, expressivity) is not well understood. Analyze the properties of this problem or technique.

Information about the project template and the source code is given below.
+ Project Paper: The project papers will be 6 pages. You can have extra pages for the references and the appendix.
They will be written in the two-column [ICML](https://icml.cc) format, using the ICML template which you can find in the corresponding website.
+ Project Source Code: Please put your source code into github and include a link in your project writeup. 
On the github page, please document exactly how to run your source code.


## Presentations
Each student will be doing 1 presentations in the term. Each presentation will be about 40 minutes long plus questions.
Here are the important points summarizing what you have to do for your presentations.

+ You must present with slides. The content in your slides should be your own but you can use others' materials, e.g., 
figures from the paper we are reading, when necessary and by crediting your source on your slide.
+ Please have a separate slide for each of 4 questions in the summary item in the Paper Review section.
+ It is very helpful to demonstrate the ideas in the paper through examples. So try to have examples in your presentation.
