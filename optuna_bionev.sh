#!/bin/bash

bionev --input ./data/DrugBank_DDI/DrugBank_DDI.edgelist \
         --output ./embeddings/ProNetMF_DrugBank_DDI.txt \
         --method ProNetMF \
         --task link-prediction \
         --eval-result-file eval_result.txt
