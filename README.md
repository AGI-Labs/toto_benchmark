# Train Offline, Test Online: A Real Robot Learning Benchmark
<!-- TODO: add teaser figures, some setup/task images, etc  -->
![toto_dataset](docs/images/toto_dataset.gif)

## Prerequisites
- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)

## Installation
You can either use a local conda environment or a docker environment.

### Setup conda environment
1. Run the following command to create a new conda environment: ```source setup_toto_env.sh```

### Setup docker environment
1. Follow the instructions in [here](https://github.com/AGI-Labs/toto_benchmark/blob/main/docker/README.md).

Note: If you are contributing models to TOTO, we strongly suggest setting up the docker environment.

### TOTO Visual Representation Models
### TOTO Datasets
<!-- TODO: need to update the dataset link after google drive clean up -->
TOTO consists of two tabletop manipulations tasks, scooping and pouring. The datasets of the two tasks can be downloaded [here](https://drive.google.com/drive/folders/1JGPGjCqUP4nUOAxY3Fpx3PjUQ_loo7fc?usp=share_link).

*Update*: please download the scooping data from Google Cloud Bucket [here](https://console.cloud.google.com/storage/browser/toto-dataset) instead.

<!-- TODO: update link to dataset README.md file. May consider create a dataset/ folder and add the readme into the repo -->
We release the following datasets: 
- `cloud-dataset-scooping.zip`: TOTO scooping dataset
- `cloud-dataset-pouring.zip`: TOTO pouring dataset

Additional Info:
- `scooping_parsed_with_embeddings_moco_conv5.pkl`: the same scooping dataset parsed with MOCO (Ours) pre-trained visual representations. (included as part of the TOTO scooping dataset) 
- `pouring_parsed_with_embeddings_moco_conv5.pkl`: the same pouring dataset parsed with MOCO (Ours) pre-trained visual representations. 
(included as part of the TOTO pouring dataset)

For more detailed dataset format information, see `assets/README.md`

## Train a TOTO Behavior Cloning (BC) Agent
Here's an example command to train an image-based BC agent with MOCO (Ours) as the image encoder. You will need to download `scooping_parsed_with_embeddings_moco_conv5_robocloud.pkl` to have this launched.

```
cd toto_benchmark
 
python scripts/train.py --config-name train_bc.yaml data.pickle_fn=../assets/cloud-dataset-scooping/scooping_parsed_with_embeddings_moco_conv5_robocloud.pkl
```

<!-- TODO: instructions on training agents with other vision representations? need to parse the dataset, etc -->

## Contributing to TOTO

 We invite the community to submit their methods to TOTO benchmark. We support the following challenges:

- **Challenge 1**: a pre-trained visual representation model. 
- **Challenge 2**: an agent policy which uses either a custom visual representation or the ones we provide.

### Challenge 1: Visual Representation Model Challenge

To submit your custom visual representation model to TOTO, you will train your visual representation model in any preferred way, generate image embeddings for TOTO datasets with your model, and finally train and submit a BC agent on this dataset. You will submit both your visual representation model and the BC model, as your visual representation model will be loaded and called during evaluations. We have provided scripts for interfacing with your vision model and for BC training. Please see the following instructions for details. 

*Note that you may or may not use TOTO datasets when training your visual representation model.*

<!-- TODO: mention somewhere the assumption that our BC pipeline assume your image embedding to be a 1D vector? -->
- Download the datasets [here](https://drive.google.com/drive/folders/1JGPGjCqUP4nUOAxY3Fpx3PjUQ_loo7fc?usp=share_link).
- Move and unzip the datasets in `assets/`, so it has the following structure:
    ```
    assets/
    - cloud-dataset-scooping/
        - data/
    - cloud-dataset-pouring/
        - data/
    ```
- Train the model in your preferred way. If you plan to train it on our datasets, feel free to use the provided `assets/example_load.py` for loading images in our datasets.
<!-- TODO: add example_load.py to github, and update this with a link -->

- After your model is trained, update the file `toto_benchmark/vision/CollaboratorEncoder.py` for loading your model, as well as any transforms needed. Feel free to include additional files needed by your model under `toto_benchmark/vision/`.
    - The functions defined in `CollaboratorEncoder.py` take in a config file. An example config file `toto_benchmark/conf/train_bc.yaml` has been provided. You may specify `vision_model_path` in the config file for loading your model, as well as add additional keys if needed. You will use this config when training a BC agent later.
    - Please make sure your vision model files are outside of `assets/`, as it will be ignored when generating files for submission later.

- Update your model's embedding size in `toto_benchmark/vision/__init__.py`.
- Launch `data_with_embeddings.py` to generate a dataset with image embeddings generated by your model. 

    ```
    # Example command for the scooping dataset: 
    cd toto_benchmark

    python scripts/data_with_embeddings.py --data_folder ../assets/cloud-dataset-scooping/ --vision_model collaborator_encoder 
    ```
    After this, a new data file `assets/cloud-dataset-scooping/parsed_with_embeddings_collaborator_encoder.pkl` will be generated. 
- Now we are ready to train a BC agent! Here's an example command for training with config `train_bc.yaml`:
    ```
    python scripts/train.py --config-name train_bc.yaml data.pickle_fn=../assets/cloud-dataset-scooping/parsed_with_embeddings_collaborator_encoder.pkl agent.vision_model='collaborator_encoder'
    ```
    A new agent folder will be created in `outputs/<path_to>/<agent>/`.
- Once the above is done, run `python scripts/test_stub_env.py -f outputs/<path_to>/<agent>/` for a simple simulated test on the robot. If everything works as expected, we are ready to have the agent to be evaluated on the real robot!
- For submission, Run ```prepare_submission.sh``` script to generate a zipped folder which is ready for submission.

### Challenge 2: Agent Policy Challenge
To submit your agent, you will train your image-based agent on our datasets in any preferred way. You may develop your custom visual representation model or use existing ones in TOTO. Please see below for detailed instructions: 
- Download the datasets [here](https://drive.google.com/drive/folders/1JGPGjCqUP4nUOAxY3Fpx3PjUQ_loo7fc?usp=share_link) and train your agents in your preferred way.
- *(Optional)* If you plan to use any existing TOTO visual representation model, we release the pre-trained models [here](https://drive.google.com/drive/folders/1iqDIIIalTi3PhAnFjZxesksvFVldK42p?usp=sharing). Download the models and put them into `assets/`. Then, simply use our provided functions to load the models as follows:
    ```
    from toto_benchmark.vision import load_model, load_transforms
    img_encoder = load_model(config)
    transforms = load_transforms(config)
    
    # See conf/train_bc.yaml for an example config file.
    ``` 
- Update the agent file: `toto_benchmark/agents/CollaboratorAgent.py`. This acts as a wrapper around your model to interface with our robot stack. Please refer to `toto_benchmark/agents/Agent.py` for more information
- Update the agent config file `toto_benchmark/outputs/collaborator_agent/hydra.yaml` for initializing your agent
- Once the above is done, run 
    ```
    cd toto_benchmark

    python scripts/test_stub_env.py -f outputs/collaborator_agent/
    ``` 
for a simple simulated test on the robot. If everything works as expected, we are ready to have the agent to be evaluated on the real robot!
- For submission, Run ```prepare_submission.sh``` script to generate a zipped folder which is ready for submission.
    - Please make sure your agent files are outside of `assets/`, as it will be ignored for submission.
