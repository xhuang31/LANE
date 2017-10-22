# Label Informed Attributed Network Embedding
Label Informed Attributed Network Embedding, WSDM 2017

## Code in MATLAB
```
H = LANE_fun(Net,Attri,LabelY,d,alpha1,alpha2,numiter);  
H = AANE_fun(Net,Attri,d,alpha1,alpha2,numiter);  
```

- H is the joint embedding representation of Net and Attri;
- Net is the weighted adjacency matrix;
- Attri is the node attribute information matrix with row denotes nodes;
- LabelY is the label information.
 
## Reference in BibTeX:
@conference{Huang-etal17Label,  
Author = {Xiao Huang and Jundong Li and Xia Hu},  
Booktitle = {ACM International Conference on Web Search and Data Mining},  
Pages = {731--739},  
Title = {Label Informed Attributed Network Embedding},  
Year = {2017}}
