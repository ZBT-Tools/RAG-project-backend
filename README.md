Build a Chroma database via running the `construct_vectorstore_mineru.ipynb` notebook. 
I was having problems with installing `mineru` within my current environment, so I had to create another Conda environment to run the notebook:

```
conda create -n MinerU
conda activate MinerU
conda install pip
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
```


You can also create the database by running the `construct_vectorstore.py` file, however the `mineru` variant has shown more success for the RAG system.

Next, simply run the `main.py` file:

```
python main.py
```

The app should run on port 5000.