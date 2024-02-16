<div align="center"><img src="https://keras.io/img/logo-small.png" alt="Keras logo" width="100"><br/>
This starter notebook is provided by the Keras team.</div>

# PII Data Detection with [KerasNLP](https://github.com/keras-team/keras-nlp) and [Keras](https://github.com/keras-team/keras)

> The objective of this competition is to detect and remove personally identifiable information (PII) from student writing.

<div align="center">
    <img src="https://i.ibb.co/3stPB0t/pii-data-detection.jpg" alt="PII Data Detection">
</div>

The task of this competition falls under **Token Classification** (not Text Classification!), sometimes known as **Named Entity Recognition (NER)**. This notebook guides you through performing this task from scratch for the competition. Implementing from scratch is a unique feature of this notebook, as most public notebooks use **HuggingFace** to handle modeling and data processing, which performs many tasks under the hood. One may have to look deeper into the repository to understand what is happening inside. In contrast, this notebook goes step by step, showing you exactly how Token Classification works. A cherry on top: this notebook leverages **Mixed Precision** and **Distributed (multi-GPU)** Training/Inference to turbocharge performance!

🔗 **Notebook** (Train + Inference): [**PII Data Detection: KerasNLP Starter Notebook**](https://www.kaggle.com/code/awsaf49/pii-data-detection-kerasnlp-starter-notebook) You can also find it in the [/notebooks](./notebooks) folder of this repository.

<u>Fun fact</u>: This notebook is backend-agnostic, supporting TensorFlow, PyTorch, and JAX. Utilizing KerasNLP and Keras allows us to choose our preferred backend. Explore more details on [Keras](https://keras.io/keras_3/).

In this notebook, you will learn how to:

- Design a data pipeline for token classification.
- Create a model for token classification with KerasNLP.
- Load the data efficiently using [`tf.data`](https://www.tensorflow.org/guide/data).
- Perform Mixed Precision and Distributed Training/Inference with Keras 3.
- Make submission on test data.

**Note**: For a more in-depth understanding of KerasNLP, refer to the [KerasNLP guides](https://keras.io/keras_nlp/).


# Data

The competition dataset contains $22,000$ student essays where $70\%$ essays are reserved for **testing**, leaving $30\%$ for **training** and **validation**.

Sure, here's the modified markdown with an example of the BIO format label:

**Data Overview:**

* All essays were written in response to the **same prompt**, applying course material to a real-world problem.
* The dataset includes **7 types of PII**: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`.
* Labels are given in **BIO (Beginning, Inner, Outer)** format.

**Example of BIO format label:**

Let's consider a sentence: `"The email address of Michael jordan is mjordan@nba.com"`. In BIO format, the labels for the personally identifiable information (PII) would be annotated as follows:

| **Word** | The | email | address | of | Michael | Jordan | is | mjordan@nba.com |
|----------|-----|-------|---------|----|---------|--------|----|----------------|
| **Label** | O   | O     | O       | O  | B-NAME_STUDENT | I-NAME_STUDENT | O  | B-EMAIL        |

In the example above, `B-` indicates the beginning of an PII, `I-` indicates an inner part of a multi-token PII, and `O` indicates tokens that do not belong to any PII.

**Data Format:**

* The train/test data is stored in `{test|train}.json` files.
* Each json file has:
    * `document`: unique ID (integer)
    * `full_text`: essay content (string)
    * `tokens`: individual words in the essay (list of strings)
    * `labels` (training data only): BIO labels for each token (list of strings)
