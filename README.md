# Episodic Memory Transformers

Note: This is a work in progress, and only serves as a proof of concept.

Augmenting Transformers with episodic memory for improved sample efficiency.

## Overview

This project is a simple proof-of-concept of attention-based episodic memory as a means of improving sample efficiency. In theory, access to previously-seen experiences will provide information regarding the solution to a new task. A model can therefore learn to recall these experiences by leveraging a learned attention-based similarity across the memory table. In practice, the task must facilitate usefulness of these memories. In this example, the task is next-word prediction, which is not quite proper to demonstrate the power of episodic memory.

Specifically, when inferencing, the model must have the exact source-target pair in memory in order to produce a correct result for this task.

However, given a weakly-supervised or a more general case, we should observe the expected result.

## Running

Simply run the ```main.py``` file.
