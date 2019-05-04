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
    git+https://github.com/huggingface/pytorch-pretrained-BERT.git
    ```
    
3. Download the English spacy model.
    ```
    python -m spacy download en
    ```

## Fine-tune BERT for classification task (could be ignored)
1. Create dataset object from raw data.
    ```
    mkdir -p dataset/classification
    python3.7 -m BERT.create_dataset dataset/classification
    ```   

2. Fine-tune the model.
    ```
    mkdir -p BERT/models/MODEL_NAME
    cp BERT/models/strong0/config.yaml BERT/models/MODEL_NAME/config.yaml
    python -m BERT.train BERT/models/MODEL_NAME
    ```
    
    Every epoch, a checkpoint of model parameters will be saved in
    `BERT/models/MODEL_NAME/ckpts`.

    You can observe training log with
    ```
    tail -f BERT/models/MODEL_NAME/log.csv
    ```

    If you ran into GPU out-of-memory error, you can increase the value of
    `train.n_gradient_accumulation_steps` to reduce the memory usage. This may make the
    training process a bit slower, but the performance should not be affected too much.

    If you want to train another model, simply repeat the above process with a different
    `MODEL_NAME`. Note that if the `BERT/models/MODEL_NAME` directory contains `ckpts/` or
    `log.csv`, the training script will not continue in case of overwriting existing
    experiment. 

3. Make prediction.
    Based on the development set performance, you can choose which epoch's model checkpoint to use to generate prediction. Besides, you can choose multiple models to test the ensemble. Optionally, you can specify the batch size.
    ```
    python -m BERT.predict BERT/models/MODEL_NAME0/epoch-EPOCH.ckpt [BERT/models/MODEL_NAME1/epoch-EPOCH.ckpt] PREDICTION_DIR_PATH --batch_size BATCH_SIZE
    ```
    You will then have a prediction as a csv file under
    `PREDICTION_DIR_PATH/`.

## Test BERT for classification task
1. Download the pre-fine-tuned BERT for classification task models.
    ```
    bash download.sh
    ```
    
2. Test the ensemble that passes the strong baseline.
    ```
    bash strong.sh TEST_CSV_PATH PREDICTION_DIR_PATH
    ```

3. Test the ensemble that has the best performance.
    ```
    bash best.sh TEST_CSV_PATH PREDICTION_DIR_PATH
    ```