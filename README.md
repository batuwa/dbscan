# Simple DBSCAN Algorithm

A basic implementation of the [Original DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN#Original_Query-based_Algorithm). 
It mimicks `scikit-learn`'s `model.fit()` API.

![DBSCAN clusters](https://github.com/batuwa/dbscan/blob/master/plots/clusters.png)

The output is fairly close to `scikit-learn`'s built in DBSCAN implementation.  

For comparison look at the `notebooks` folder.

## Requirements

You need Python 3.7 or later since this library uses the [dataclasses](https://docs.python.org/3/library/dataclasses.html) feature introduced in Python 3.7.

## Testing

Run test using the command

```bash
python -m unittest test
```
