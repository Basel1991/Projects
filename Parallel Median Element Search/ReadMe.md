In this course project, parallel processing was used to search for the median in a huge integer arrays. One-million-element array took 16 ms only to find the median. 

Methodology:
First, the array is divided to sub-arrays which are sorted in parallel. Then, they are merged in parallel as well. Third and last, the element in the middle is picked (the median). 
