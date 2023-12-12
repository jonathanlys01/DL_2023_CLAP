
# Usage
### the ```main.py``` script
The ```main.py``` script is the main script of the project. It allows inference (zero-shot classification) over the selected dataset.

To run the script, you need to specify the following arguments :
- ```--dataset``` (or ```-d```) : the dataset to use. Possible values are ```esc50```, ```urbansound8k``` and ```fma```. (shortcuts : ```e```, ```u``` and ```f```)
- ```--limit``` (or ```-l```) : the number of samples to use. (default : ```-1```, which means all the samples)
- ```--plot``` (or ```-p```) : what to plot. Possible values are ```no```, ```cm```, ```audio``` and ```all```. (default : ```no```)
- ```--model``` (or ```-m```) : the model to use. Possible values are ```default```, ```general``` and ```music```. ("laion/clap-htsat-unfused" for general and "laion/larger_clap_music" for music). Default is will automatically choose the model depending on the dataset. (default : ```default```)
- ```--topk``` (or ```-k```) : the number of top classes to use for the accuracy and the confusion matrix. (default : ```1``` i.e. argmax of similarity) 
-  ```--verbose``` (or ```-v```) : whether to print the progress or not. (default : ```False```)

### the ```sound_search.py``` script
The ```sound_search.py``` script allows to search for a sound in a given dataset (or all the datasets) from a text query.

To run the script, you need to specify the following arguments :
- ```--dataset``` (or ```-d```) : the dataset to use. Possible values are ```esc50```, ```urbansound8k``` and ```fma``` or ```all```. (shortcuts : ```e```, ```u```, ```f``` and ```a```)
- ```--query``` (or ```-q```) : the query to search for.
- ```--limit``` (or ```-l```) : the number of samples to use. (default : ```-1```, which means all the samples)
- ```--model``` (or ```-m```) : the model to use. Possible values are ```general``` and ```music```. ("laion/clap-htsat-unfused" for general and "laion/larger_clap_music" for music).(default : ```general```)

### ```features_gen.py``` and ```features_viz.py```
The ```features_gen.py``` script allows to generate the features of a given dataset (or all the datasets) and save them in a ```.npz``` file.
The ```features_viz.py``` script allows to visualize the features of a given dataset (or all the datasets) from a ```.npz``` file, using t-SNE or PCA.

### ```sound_search_app.py```

This script launches a Gradio app that allows to search for a sound in a given dataset from a text query. It is faster than ```sound_search.py``` because it loads the model only once. Visualizations are also available, and use a background thread to project the features in 2D, allowing to search for a sound while the features are being projected.
