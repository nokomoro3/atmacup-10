# INSTALL

## activate venv
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $
```

## install packages
```
(.venv) $ pip3 install -r requirements.txt
```

# install of fastText
- install package in this command.
```bash
(.venv) $ git clone https://github.com/facebookresearch/fastText.git
(.venv) $ pip3 install fastText
(.venv) $ rm -rf fastText
```

- and download a model in this command.
```bash
(.venv) $ wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

# if you need, install jupyter
```bash
(.venv) $ sudo apt install jupyter-notebook
```
