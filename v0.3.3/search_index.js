var documenterSearchIndex = {"docs":
[{"location":"intro-graph/#Graph-embedding","page":"Graph embedding","title":"Graph embedding","text":"","category":"section"},{"location":"intro-graph/#Prerequisites","page":"Graph embedding","title":"Prerequisites","text":"","category":"section"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"You need to install the following packages for this demo","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using Pkg\nPkg.add([\"DataDeps\", \"MatrixMarket\"])","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"We will work with the optdigits_10NN graph, from SuiteSparse Matrix Collection. First, we will define a data dependency, to download and open the MTX file. ","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using DataDeps, MatrixMarket\n\nregister(DataDep(\"optdigits\",\n               \"\"\"\n       ML_Graph: adjacency matrices from machine learning datasets, Olaf\n       Schenk.  D.  Pasadakis,  C.  L.  Alappat,  O.  Schenk,  and  G.\n       Wellein, \"K-way p-spectral clustering on Grassmann manifolds,\" 2020.\n       https://arxiv.org/abs/2008.13210\n\n       Graph: optdigits_10NN Classes: 10\n\n       See https://sparse.tamu.edu/ML_Graph/optdigits_10NN\n               \"\"\",\n               \"https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/optdigits_10NN.tar.gz\",\n               \"336949bf9a2ac3a0643a8d7a2217792f52e0fdbec54e1b870079f257f200abfc\",\n               post_fetch_method = unpack\n))\n\nA = MatrixMarket.mmread( datadep\"optdigits/optdigits_10NN/optdigits_10NN.mtx\" );\nL = vec( MatrixMarket.mmread( datadep\"optdigits/optdigits_10NN/optdigits_10NN_label.mtx\" ) );\nL = Int.( L );","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using DataDeps, MatrixMarket\n\nregister(DataDep(\"optdigits\",\n               \"\"\"\n       ML_Graph: adjacency matrices from machine learning datasets, Olaf\n       Schenk.  D.  Pasadakis,  C.  L.  Alappat,  O.  Schenk,  and  G.\n       Wellein, \"K-way p-spectral clustering on Grassmann manifolds,\" 2020.\n       https://arxiv.org/abs/2008.13210\n\n       Graph: optdigits_10NN Classes: 10\n\n       See https://sparse.tamu.edu/ML_Graph/optdigits_10NN\n               \"\"\",\n               \"https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/optdigits_10NN.tar.gz\",\n               \"336949bf9a2ac3a0643a8d7a2217792f52e0fdbec54e1b870079f257f200abfc\",\n               post_fetch_method = unpack\n))\n\nA = MatrixMarket.mmread( datadep\"optdigits/optdigits_10NN/optdigits_10NN.mtx\" );\nL = vec( MatrixMarket.mmread( datadep\"optdigits/optdigits_10NN/optdigits_10NN_label.mtx\" ) );\nL = Int.( L );","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"We will now embed the graph using SG-t-SNE-Π","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using SGtSNEpi, Random\n\n# reproducible results\nRandom.seed!(0);\nY0 = 0.01 * randn( size(A,1), 2 );\n\nY = sgtsnepi(A; Y0 = Y0);","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using SGtSNEpi, Random\n\n# reproducible results\nRandom.seed!(0);\nY0 = 0.01 * randn( size(A,1), 2 );\n\nY = sgtsnepi(A; Y0 = Y0);","category":"page"},{"location":"intro-graph/#Visualization","page":"Graph embedding","title":"Visualization","text":"","category":"section"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"To reproduce the next steps, we need to install the following packages","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"Pkg.add([\"CairoMakie\", \"Colors\", \"Makie\"])","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"If Makie was not installed when SGtSNEpi was loaded, you need to restart Julia and repeat the previous steps.","category":"page"},{"location":"intro-graph/#Show-embedding","page":"Graph embedding","title":"Show embedding","text":"","category":"section"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"Finally, we show the embedding, using the provided visualization function","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using CairoMakie, Colors, LinearAlgebra\n\nshow_embedding( Y, L ; A = A, res = (2000, 2000) )","category":"page"},{"location":"intro-graph/#D-embedding","page":"Graph embedding","title":"3D embedding","text":"","category":"section"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"SG-t-SNE-Π enables 3D embedding as well","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using Colors\n\ncmap = distinguishable_colors(\n           maximum(L) - minimum(L) + 1,\n           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)\n\nRandom.seed!(0);\nY0 = 0.01 * randn( size(A,1), 3 );\n\nY = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);\n\nsc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 2 )\n\nrecord(sc, \"sgtsnepi-animation.gif\", range(0, 1, length = 24*8); framerate = 24) do ang\n  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )\nend","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"using Colors\n\ncmap = distinguishable_colors(\n           maximum(L) - minimum(L) + 1,\n           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)\n\nRandom.seed!(0);\nY0 = 0.01 * randn( size(A,1), 3 );\n\nY = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);\n\nsc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 2 )\n\nrecord(sc, \"sgtsnepi-animation.gif\", range(0, 1, length = 24*8); framerate = 24) do ang\n  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )\nend","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"(Image: sgtsnepi animation)","category":"page"},{"location":"intro-graph/#Clean-up","page":"Graph embedding","title":"Clean-up","text":"","category":"section"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"Remove the data by issuing","category":"page"},{"location":"intro-graph/","page":"Graph embedding","title":"Graph embedding","text":"rm(datadep\"optdigits\"; recursive=true)","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"using CairoMakie, FLANN\nCairoMakie.activate!()","category":"page"},{"location":"intro-point-cloud/#Point-cloud-data-embedding","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"","category":"section"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We provide a case study, using the MNIST dataset, to get you started with SG-t-SNE-Π. We assume you have Julia and SGtSNEpi installed already.","category":"page"},{"location":"intro-point-cloud/#Prerequisites","page":"Point-cloud data embedding","title":"Prerequisites","text":"","category":"section"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"You need to install the following packages for this demo","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"using Pkg\nPkg.add([\"MLDatasets\", \"ImageFeatures\", \"Random\", \"Images\"])","category":"page"},{"location":"intro-point-cloud/#Embedding-handwritten-digits-(MNIST-data)","page":"Point-cloud data embedding","title":"Embedding handwritten digits (MNIST data)","text":"","category":"section"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"The MNIST dataset comprises of 60000 training and 10000 testing images of handwritten digits. We shall embed the total of 70000 handwritten images.","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"First, we download the dataset","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"using SGtSNEpi, MLDatasets, ImageFeatures, Random, Images\n\nX, L = MNIST.traindata(Float64);\nX = cat( X, MNIST.testdata(Float64)[1] ; dims = 3 );\nL = cat( L, MNIST.testdata(Float64)[2] ; dims = 1 );\n\nL = Int.( vec( L ) );  # make sure labels is an integer vector\n\nn = size( X, 3 );\n\nX = permutedims( X, [2, 1, 3] );\n\nnothing; # hide","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We visualize some of the digits that appear in the data set","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"mosaicview( Gray.(X[:,:,1:600]), ncol=30, rowmajor=true )","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We transform the pixel values to Histogram of Oriented Gradients (HOG) descriptors","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"F = zeros( n, 324 );\n\nfor img = 1:n\n  F[img,:] = create_descriptor( X[:,:,img], HOG(; cell_size = 7) )\nend","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We initialize (randomly) the coordinates in the 2D embedding space (this step is crucial for reproducible results)","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"Random.seed!(0);\nY0 = 0.01 * randn( n, 2 );\nnothing; # hide","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We use SG-t-SNE-Π to embed the data in a 2D space","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"A = pointcloud2graph( F )\nY = sgtsnepi(A; Y0 = Y0);","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"A = pointcloud2graph( F )\nY = sgtsnepi(A; Y0 = Y0);","category":"page"},{"location":"intro-point-cloud/#Visualization","page":"Point-cloud data embedding","title":"Visualization","text":"","category":"section"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"To reproduce the next steps, we need to install the following packages","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"Pkg.add([\"CairoMakie\", \"Colors\", \"Makie\"])","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"If Makie was not installed when SGtSNEpi was loaded, you need to restart Julia and repeat the previous steps.","category":"page"},{"location":"intro-point-cloud/#Visualization-2","page":"Point-cloud data embedding","title":"Visualization","text":"","category":"section"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"We visualize the 70000 digits on the 2D embedding space, colored by their class. For this purpose, we use the routine vis_embedding.","category":"page"},{"location":"intro-point-cloud/","page":"Point-cloud data embedding","title":"Point-cloud data embedding","text":"\nshow_embedding( Y, L; res = (2000, 2000) )","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"(Image: SG-t-SNE-Π)","category":"page"},{"location":"#SG-t-SNE-Π:-Swift-Graph-Embedding","page":"Overview","title":"SG-t-SNE-Π: Swift Graph Embedding","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"We provide a Julia interface, i.e., a wrapper to SG-t-SNE-Π, which is a high-performance software for swift embedding of a large, sparse graph G(VEP) into a d-dimensional space (d = 123) on a shared-memory computer. The algorithm SG-t-SNE and the software t-SNE-Π were first described in Reference (Nikos Pitsianis, Alexandros-Stavros Iliopoulos, Dimitris Floros, Xiaobai Sun (2019)) and released on GitHub in June 2019 (Nikos Pitsianis, Dimitris Floros, Alexandros-Stavros Iliopoulos, Xiaobai Sun (2019)). In fact, the software admits two types of data at input: (i) a graph or network, described in terms of vertices and edges (pairwise relationships among vertices), directed or undirected; (ii) a point-cloud data set, described in terms of feature attributes in numerical values (coordinates), in a metric space. In the second case, the point-cloud data are converted into a kNN graph, in which the data points are vertices, the edges represent k-nearest-neighbor relationships. In other words, the embedding acts as a non-linear dimension reduction. The software renders 2D or 3D embedding of the vertices or the data points. Embedding in 3D allows one to observe connections or disconnections that are obscured in 2D embedding.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"SG-t-SNE improves upon two main precursors, t-SNE by Laurens van der Maaten, Geoffrey E. Hinton (2008), Laurens van der Maaten (2014) and FI-t-SNE by George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger (2019), in several aspects. (i) It removes the limitation to regular-degree graphs with t-SNE; (ii) it enables 3D embedding while following the basic method used in FI-t-SNE for calculating repulsive forces; (iii) it surpasses in eﬀiciency, the precursor algorithms and implementations on modern laptop/desktop computers, without compromising accuracy. Recently, SG-t-SNE-Π has been compared favorably to UMAP (Parashar Dhapola, Johan Rodhe, Rasmus Olofzon, Thomas Bonald, Eva Erlandsson, Shamit Soneji, G{\\\"o}ran Karlsson (2021)). This version makes SG-t-SNE readily deployable to the Julia ecosystem. More information can be found at SG-t-SNE-Π.","category":"page"},{"location":"#Installation","page":"Overview","title":"Installation","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"To install SG-t-SNE-Π through Julia, issue","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"] add SGtSNEpi","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"We provide two use cases:","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Point cloud data embedding Point-cloud data embedding\nGraph embedding Graph embedding","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"For full documentation of the functions exported by SG-t-SNE-Π, see the API","category":"page"},{"location":"#References","page":"Overview","title":"References","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"","category":"page"},{"location":"API/#API","page":"API (Advanced)","title":"API","text":"","category":"section"},{"location":"API/","page":"API (Advanced)","title":"API (Advanced)","text":"CurrentModule = SGtSNEpi","category":"page"},{"location":"API/","page":"API (Advanced)","title":"API (Advanced)","text":"SGtSNEpi.jl exports the following functions","category":"page"},{"location":"API/","page":"API (Advanced)","title":"API (Advanced)","text":"","category":"page"},{"location":"API/","page":"API (Advanced)","title":"API (Advanced)","text":"Modules = [SGtSNEpi]\nPages = [\"sgtsne.jl\", \"vis.jl\"]","category":"page"},{"location":"API/#SGtSNEpi.pointcloud2graph","page":"API (Advanced)","title":"SGtSNEpi.pointcloud2graph","text":"pointcloud2graph( X::AbstractMatrix, u = 10, k = 3*u; knn_type )\n\nConvert a point-cloud data set X (coordinates) of size N times D to a similarity graph, using perplexity equalization, same as conventional t-SNE.\n\nSpecial options for point-cloud data embedding\n\nu=10: perplexity\nk=3*u: number of nearest neighbors (for kNN formation)\nknn_type=( size(A,1) < 10_000 ) || !USING_FLANN ? :exact : :flann: Exact or approximate kNN\n\n\n\n\n\n","category":"function"},{"location":"API/#SGtSNEpi.sgtsnepi-Tuple{Graphs.AbstractGraph}","page":"API (Advanced)","title":"SGtSNEpi.sgtsnepi","text":"sgtsnepi( A::AbstractMatrix )\nsgtsnepi( G::AbstractGraph )\n\nCall SG-t-SNE-Π on the input graph, given as either a sparse adjacency matrix A or a graph object G. Alternatively, the input can be a point-cloud data set X (coordinates) of size N times D, i.e.,\n\nsgtsnepi( X::AbstractMatrix )\n\nOptional arguments\n\nd=2: number of dimensions (embedding space)\nλ=10: SG-t-SNE scaling factor\nversion=SGtSNEpi.NUCONV_BL: the version of the algorithm for computing  repulsive terms. Options are\nSGtSNEpi.NUCONV_BL (default): band-limited, approximated via  non-uniform convolution\nSGtSNEpi.NUCONV: approximated via non-uniform convolution (higher  resolution than SGtSNEpi.NUCONV_BL, slower execution time)\nSGtSNEpi.EXACT: no approximation; quadratic complexity, use only with  small datasets\n\nMore options (for experts)\n\nmax_iter=1000: number of iterations\nearly_exag=250: number of early exageration iterations\nalpha=12: exaggeration strength (applicable for first early_exag iterations)\nY0=nothing: initial distribution in embedding space (randomly generated if nothing). You should set this parameter to generate reproducible results.\neta=200.0: learning parameter\ndrop_leaf=false: remove edges connecting to leaf nodes\n\nAdvanced options (performance-related)\n\nnp=0: number of threads (set to 0 to use all available cores)\nh=1.0: grid side length\nlist_grid_size = filter( x -> x == nextprod( (2, 3, 5), x ), 16:512 ):  the list of allowed grid size along each dimension. Affects FFT performance;  most efficient if the size is a product of small primes.\nprofile=false: disable/enable profiling. If enabled the function  return a 3-tuple: (Y, t, g), where Y is the embedding  coordinates, t are the execution times of each module per iteration  (size 6 x max_iter) and g contains the grid size, the  embedding domain size (maximum(Y) - minimum(Y)), and the scaling factor  s_k for the band-limited version, per dimension (size 3 x max_iter).\n\nNotes\n\nIsolated nodes are placed randomly on the top-right corner of the embedding space\nThe function tries to automatically detect whether the input matrix represents an adjacency matrix or data coordinates. In ambiquous cases, such as a square matrix of data coordinates, the user may specify the type using the optional argument type\n:graph: the input is an adjacency matrix\n:coord: the input is the data coordinates\n\nExamples\n\njulia> using LightGraphs\n\njulia> G = circular_ladder_graph( 500 )\n{1000, 1500} undirected simple Int64 graph\n\njulia> Y = sgtsnepi( G; np = 4, early_exag = 100, max_iter = 250 );\nNumber of vertices: 1000\nEmbedding dimensions: 2\nRescaling parameter λ: 10\nEarly exag. multiplier α: 12\nMaximum iterations: 250\nEarly exag. iterations: 100\nLearning rate: 200\nBox side length h: 1\nDrop edges originating from leaf nodes? 0\nNumber of processes: 4\n1000 out of 1000 nodes already stochastic\nm = 1000 | n = 1000 | nnz = 3000\nWARNING: Randomizing initial points; non-reproducible results\nSetting-up parallel (double-precision) FFTW: 4\nIteration 1: error is 96.9204\nIteration 50: error is 84.9181 (50 iterations in 0.039296 seconds)\nIteration 100: error is 4.32754 (50 iterations in 0.038005 seconds)\nIteration 150: error is 2.54655 (50 iterations in 0.066491 seconds)\nIteration 200: error is 1.90124 (50 iterations in 0.159556 seconds)\nIteration 249: error is 1.65057 (50 iterations in 0.213149 seconds)\n --- Time spent in each module ---\n\n Attractive forces: 0.006199 sec [1.24082%] |  Repulsive forces: 0.49339 sec [98.7592%]\n\n\n\n\n\n","category":"method"},{"location":"API/#SGtSNEpi.neighbor_recall-Tuple{Any, Any}","page":"API (Advanced)","title":"SGtSNEpi.neighbor_recall","text":"neighbor_recall(X, Y[; k = 10])\n\nHistogram of the stochastic k-neighbors recall values at all vertices.\n\nrm recall(v) = sum_j mathbfP_ji b_ij\n\nwhere mathbfB_k = b_ij is the adjacency matrix of the kNN graph of the embedded points mathbfY in Euclidean distance, and mathbfP_ji is the neighborhood probability matrix in the high-dimensional space mathbfX.\n\nSee Fig.4 in https://arxiv.org/pdf/1906.05582.pdf for more details.\n\n\n\n\n\n","category":"method"},{"location":"API/#SGtSNEpi.show_embedding","page":"API (Advanced)","title":"SGtSNEpi.show_embedding","text":"show_embedding( Y [, L] )\n\nVisualization of 2D embedding coordinates Y using Makie. If the n times 1 vector L of vertex memberships is provided, points are colored accroding to the labels.\n\nRequirements\n\nThis function is provided only if the user already has Makie installed in their Julia environment. Otherwise, the function will not be defined.\n\nNotes\n\nYou need to install and select a backend for Makie. See instructions by Makie for more details. For example, to show figures in an interactive window, issue\n\n] add GLMakie\nusing GLMakie\nGLMakie.activate!()\n\nOptional arguments (for experts)\n\nA=nothing: adjacency matrix; if provided, edges are shown\ncmap: the colormap to use (default to distinguishable colors)\nres=(800,800): figure resolution\nlwd_in=0.5: line width for internal edges\nlwd_out=0.3: line width for external edges\nedge_alpha=0.2: the alpha channel for the edges\nclr_in=nothing: set color for all intra-cluster edges (if nothing, color by cmap)\nclr_out=colorant\"#aabbbbbb\": the color of inter-cluster edges\nmrk_size=4: marker size\nsize_label:24: legend label size\n\n\n\n\n\n","category":"function"}]
}
