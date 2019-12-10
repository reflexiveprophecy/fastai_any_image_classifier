## Fast AI Any Image Classifier

This is a project that helps to efficiently streamline the process of collecting image datasets and building an image classifier using the FastAI library. 

Warning: We are not responsible for any copyright issues associated with the images collected from Google Images. 

After git clone the repository, please run the following commands to be able to use the project:

    python3 -m venv venv
    pip install -r requirements.txt

Please note that after .lr_find() is run, a learning rate graph will be generated and named as learning_rate_graph.png in the data directory, you would need to manually input the range of desired learning rates based on where losses descend the fastest on the chart, for the following 'unfrozen'
training process. 

