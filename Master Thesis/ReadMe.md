![thesis_readme](https://user-images.githubusercontent.com/23275312/59156695-de912c80-8a9f-11e9-8c54-c02afaf69d08.png)

# Master Thesis
This is where I put code + utilities related to my Master's thesis in Erasmus Mundus in Medical Imaging and Applications.
## Generative Adversarial Networks (GANs) for Realistic Data Augmentation and Lesion Simulation in X-ray Breast Imaging.
I this work DCGAN was used to generate mammographic patches (128 X 128 pixels).

### DCGAN Training.

The following Figure shows the process of training the DCGAN.

![GAN](https://user-images.githubusercontent.com/23275312/59156752-b950ee00-8aa0-11e9-9b3a-03c83ea8387a.png)


### Results

In this section we show two batches, one real and one fake to be able to compare the quality of the generated images.

![real_fake_2](https://user-images.githubusercontent.com/23275312/59156914-367d6280-8aa3-11e9-98c9-7ce785905fcb.png)



### Evaluation

#### Trained Generators for Augmenting Unbalanced Classification Problems
In this figure we evaluate the performance of using the DCGAN as an augmentation tool as a function of the trinaing size.

![f1score_allfolds](https://user-images.githubusercontent.com/23275312/59156842-5a8c7400-8aa2-11e9-9432-33dcd9d2b2ad.png)


#### t-Stochastic Neighbors Embedding (t-SNE) Analysis

In this section we compare the distribution of the generated images and the real ones. We show that the synthetic images support the real ones by filling the gaps in the distribution which justifies previous results where DCGAN helped to increase the generalisation of the classsifier.


![tSNE_real_fake](https://user-images.githubusercontent.com/23275312/59156947-ab509c80-8aa3-11e9-8495-1dec1468c3d6.png)


### Acknowledgments

Our great gratitude goes to Nvidia for supporting this work by a Titan X GPU. I would like also to thank my supervisors for offering decent infrastructure and dataset as well as directing me to the target. I am in huge debt to Vicorob research institute, especially Mostafa Abubakr Salem (PhD) for his fruitful discussions and valuable suggestions regarding DCGAN training and testing. Special thanks go to Lavsen Dahal (MAIA) and Albert Garcia (PhD) for their continuous support and enlightening insights.

![MAIA](https://user-images.githubusercontent.com/23275312/59156972-ddfa9500-8aa3-11e9-8891-560f0ce716a1.jpg)
