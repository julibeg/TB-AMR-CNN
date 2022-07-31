# TB-AMR-CNN

Neural Network inspired by [MTB-CNN](https://github.com/aggreen/MTB-CNN) to
predict antibiotic resistance in _M. tuberculosis_. The defining feature of this
implementation is that it is fully independent of the input dimensions while
showing similar performance to the original. Therefore, new sequences don't need
to be aligned to a fixed-length MSA (which would drop any insertions).

The trained model was packaged in a [Docker container](https://hub.docker.com/repository/docker/julibeg/tb-ml-neural-net-predictor-13-drugs)
for easy use (or to be used in [tb-ml](https://github.com/jodyphelan/tb-ml)).
It takes one-hot-encoded sequences of 18 loci as input and predicts resistance
against 13 drugs (amikacin, capreomycin, ciprofloxacin, ethambutol, ethionamide,
isoniazid, kanamycin, levofloxacin, moxifloxacin, ofloxacin, pyrazinamide,
rifampicin, streptomycin). There is an accompanying pre-processing container for
exracting the one-hot-encoded sequences from a SAM/BAM/CRAM file with reads aligned
against the H37Rv reference. For details on the containers please refer to their
repositories (for the 
[neural net](https://github.com/julibeg/tb-ml-containers/tree/main/neural_net_predictor_13_drugs)
or the [pre-processing pipeline](https://github.com/julibeg/tb-ml-containers/tree/main/one_hot_encode)).
