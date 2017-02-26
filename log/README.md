
Logs/Output
===========

Currently the main.lua script uses the symbolic link tnn_train/bin/dataDst as the location to store results/logs etc.
Within this then, each result dir is categorized by model architecture.

Since this data can be significant in size, you may wish to redirect this symbolic link.

1. Delete existing link:
```
~$ cd tnn_train/bin
~$ rm dataDst
```

2. Add your new link
```
~$ cd lsr/bin
~$ ln -s </your/data/dir> dataDst
```


Querying result data
--------------------

Within a particular log output directory, a results.json will record network performance results, and possibly trace data.
The Apache DRILL https://drill.apache.org standard SQL query engine support the query/analysis of this semi-structured data.

An example of the results JSON model is below.

Example query for test result values:
```
select tbl.epoch as epoch, tbl.results.train as train, tbl.results.valid as valid, tbl.results.test as test from dfs.`/path/to/results.json` as tbl order by epoch DESC;
```

Example query for average batch loss for each epoch:
```
select tbl.epoch as epoch, avg(flatten(tbl.batches.loss)) as avg_loss from dfs.`/path/to/results.json` as tbl group by epoch;
```

Example query for listing the learning rate for each batch:
```
select flatten(tbl.batches.learningRate) as LR from dfs.`/path/to/results.json` as tbl;
```

Note: trace data is only populated when appropriate configuration is set.


Example results.json:

[{
    "epoch":1,
    "results":{
      "train":89.12,
      "valid":88.73
    },
    "batches":{
      "loss":[...],
      "learningRate":[...]
    }
},{
    "epoch":2,
    "results":{
      "train":91.07,
      "valid":90.19
    },
    "batches":{
      "loss":[...],
      "learningRate":[...]
    }
}]
