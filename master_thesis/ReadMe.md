![Github_thesis_header](https://user-images.githubusercontent.com/23275312/62536498-6a2ffb00-b84e-11e9-9100-d7c2b452a3be.png)

# Master Thesis
This is where I put code + utilities related to my Master's thesis in Erasmus Mundus in Medical Imaging and Applications.
## Generative Adversarial Networks (GANs) for Realistic Data Augmentation and Lesion Simulation in X-ray Breast Imaging.
In this work, DCGAN [1] was used to generate mammographic patches (128 X 128 pixels) that have comparable realism and diversity to real ones measured by Frechet Inception Distance [2].

### DCGAN Training.

The following Figure shows the process of training the DCGAN step by step, namely:

1. Generate a batch of latent vectors (z ~ N(0,1)).
1. Forward the batch of latent vectors through G to generate a batch of fake lesions.
1. Forward the fake lesions batch (dashed arrow) as well as the real batch (dense arrow) through D.
1. Calculate the loss of D
1. Update D weights using backpropagation.
1. Calculate G loss.
1. Update G using backpropagation.

This process is repeated until all real images are used to complete one epoch.


![GAN](https://user-images.githubusercontent.com/23275312/59156752-b950ee00-8aa0-11e9-9b3a-03c83ea8387a.png)


### Results
#### Qualitatively
In this section we show two batches, one real and one fake to be able to compare the quality of the generated images.

![real_fake_2](https://user-images.githubusercontent.com/23275312/59156914-367d6280-8aa3-11e9-98c9-7ce785905fcb.png)



### Quantitatively

#### Trained Generators for Augmenting Unbalanced Classification Problems
In this figure we evaluate the performance of using the DCGAN as an augmentation tool as a function of the trinaing size.
The four augmentation approaches investigated (see the figure below) are:
 + *ORG*: using original images, the input for the classifier is Pk as positive images plus Nk as negative.
 + *Aug ORG*: original images were augmented using random horizontal and vertical flipping.
 + *GAN*: the training set of the classifier is k real masses and 1.5 X k synthetic masses as the positive class, and 10 X k normal  tissue patches as the negative class.
 + *Aug GAN*: the 1.5 X k generated images as well as the real ones were augmented on the fly by random horizontal and vertical flipping. 
   
Because the dataset is imbalanced, F1 score was used as an evaluation metric. This provides equal importance to precision and recall. The test and validation sets were fixed for all k's. 3-fold cross validation was used to assure reliable results.

![f1score_allfolds](https://user-images.githubusercontent.com/23275312/59156842-5a8c7400-8aa2-11e9-9432-33dcd9d2b2ad.png)


#### t-Stochastic Neighbors Embedding (t-SNE) Analysis

In this section we compare the distribution of the generated images and the real ones. We show that the synthetic images support the real ones by filling the gaps in the distribution which justifies previous results where DCGAN helped to increase the generalisation of the classsifier.


![tSNE_real_fake](https://user-images.githubusercontent.com/23275312/59156947-ab509c80-8aa3-11e9-8495-1dec1468c3d6.png)

### Conclusion
To sum up, GANs are a powerful tool that can be used
to generate synthetic images to be used in a variety of
applications including augmenting unbalanced classification
problems and unlocking realistic unseen images.
However, they have to be trained carefully and better be
accompanied with traditional flipping augmentation.
### Acknowledgments

Our great gratitude goes to Nvidia for supporting this work by a Titan X GPU. I would like also to thank my supervisors for offering decent infrastructure and dataset as well as directing me to the target. I am in huge debt to Vicorob research institute, especially Mostafa Abubakr Salem (PhD) for his fruitful discussions and valuable suggestions regarding DCGAN training and testing. Special thanks go to Lavsen Dahal (MAIA) and Albert Garcia (PhD) for their continuous support and enlightening insights.

### References
[1] Radford, A., Metz, L., Chintala, S., 2016. Unsupervised Representation
Learning with Deep Convolutional Generative Adversarial
Networks, in: 2016 International Conference on Learning Representations
(ICLR).

[2] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.,
2017. GANs Trained by a Two Time-Scale Update Rule Converge
to a Local Nash Equilibrium, in: Guyon, I., Luxburg, U.V., Bengio,
S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (Eds.),
Advances in Neural Information Processing Systems 30. Curran
Associates, Inc., pp. 6626–6637. http://papers.nips.cc/paper/7240-
gans-trained-by-a-two-time-scale-update-rule.

![MAIA](https://user-images.githubusercontent.com/23275312/59156972-ddfa9500-8aa3-11e9-8891-560f0ce716a1.jpg)

_MAIA (Joint Master Degree in MedicAl Imaging and Applications) master is a 2-year joint master degree (120 ECTS) coordinated by the University of Girona (UdG, Spain) and with the University of Bourgogne (uB, France) and the Università degli studi di Cassino e del Lazio Meridionale (UNICLAM, Italy) as partners._

https://maiamaster.udg.edu/
