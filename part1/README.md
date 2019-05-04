## Requirements
1. Python==3.7

2. Install the following required packages.
    ```
    torch==1.0.1
    spacy==2.0.12
    pyyaml==5.1
    python-box==3.2.4
    tqdm==4.31.1
    ipdb==0.12
    ```

## Train ELMo from scratch
1. Prepare the corpus_tokenized.txt (pre-tokenized corpus) in `data/language_model/corpus_tokenized.txt`.

2. Preprocess the corpus to create the dataset and word/character vocabulary.
    ```    
    python3.7 -m ELMo.preprocessing dataset/language_model
    ```
    
3. Train ELMo model.
    ```    
    python3.7 -m ELMo.train ELMo/models/MODEL_NAME
    ```

## Train BCN + ELMo for classification task (could be ignored)
1. Create dataset object from raw data.
    ```
    mkdir -p dataset/classification
    cp bcn_classification_dataset_config_template.yaml dataset/classification/config.yaml
    python3.7 -m BCN.create_dataset dataset/classification
    ```
    **Do not modify the content in `config.yaml`.**

2. Train model.
    ```
    mkdir -p BCN/models/MODEL_NAME
    cp model/submission/config.yaml BCN/models/MODEL_NAME/config.yaml
    python3.7 -m BCN.train BCN/models/MODEL_NAME
    ```
    **Other than `random_seed`, `device.*`, `elmo_embedder.*`, `use_elmo`,
    `train.n_epochs` and `train.n_gradient_accumulation_steps`, do not modify other
    settings in `config.yaml`.**

    Every epoch, a checkpoint of model parameters will be saved in
    `BCN/models/MODEL_NAME/ckpts`.

    You can observe training log with
    ```
    tail -f BCN/models/MODEL_NAME/log.csv
    ```

    If you ran into GPU out-of-memory error, you can increase the value of
    `train.n_gradient_accumulation_steps` to reduce the memory usage. This may make the
    training process a bit slower, but the performance should not be affected too much.

    If you want to train another model, simply repeat the above process with a different
    `MODEL_NAME`. Note that if the `BCN/models/MODEL_NAME` directory contains `ckpts/` or
    `log.csv`, the training script will not continue in case of overwriting existing
    experiment. 

3. Make prediction.

    Based on the development set performance, you can choose which epoch's model
    checkpoint to use to generate prediction. Optionally, you can specify the batch size.
    ```
    python3.7 -m BCN.predict BCN/models/MODEL_NAME EPOCH --batch_size BATCH_SIZE
    ```
    You will then have a prediction as a csv file that can be uploaded to kaggle under
    `BCN/models/MODEL_NAME/predictions/`.

## Test BCN + ELMo for classification task
1. Download the pre-trained ELMo and BCN models.
    ```
    bash download.sh
    ```
    
2. Make sure the character vocabulary exists in `dataset/language_model/char_vocabulary.pkl`. If not, please refer to ## Training ELMo to see how to create it.

3. Make prediction.
    ```
    bash simple.sh
    ```