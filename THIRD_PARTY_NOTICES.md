# Third-party data and software

## TANVOM v1

- Title: TANVOM v1: Dataset for Orienteering Map Vectorization
- Creators: Borbás, Péter; Troll, Ede
- Source: https://doi.org/10.5281/zenodo.19437536
- Record: https://zenodo.org/records/19437536
- License recorded by the Zenodo record: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/

`assets/validation_best.png` and `assets/validation_worst.png` contain processed validation tiles and masks from this dataset, together with model probability maps, binary predictions and colored errors generated in this experiment. The source pixels have been arranged into panels; prediction and error panels are derived additions. These data-derived illustrations retain CC BY 4.0 attribution. No endorsement by the dataset creators is implied. Full original map sheets and OCAD projects are not redistributed.

TANVOM preprocessing, annotations and the official map-level split are provided by the dataset creators; they are not claimed as original work by this repository's author. See the linked dataset record for its associated research and citation information.

## Software dependencies

The model is constructed using `segmentation-models-pytorch` (MIT), which provides the U-Net implementation and ResNet18 encoder integration. The training code uses PyTorch (BSD-style license), torchvision and their dependencies. Dependency source code is not vendored in this repository. Consult each upstream project's license for its terms.

ImageNet encoder weights are downloaded by the dependency during initial training and are not committed to Git. The separately published experiment checkpoint was fine-tuned on TANVOM. This notice does not relicense third-party data, software or weights.
