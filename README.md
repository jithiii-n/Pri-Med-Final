# Pri-Med-Final
User friendly privacy preserving inference portal for medical images.

This is created as part of the final year academic project for the completion of MCA course. Pri-Med uses a custom designed CNN model for the inference. The inference pipeline is encrypted with homomorphic encryption. Only the output is exposed in the pipeline , thus providing security and privacy to patient data.


The main challenge is designing a homomorphic compatible cnn model. As of the 9th commit, the development is on going in a partially homomorphic way. The inputs are encrypted using the CKKS scheme when uploading to the uploads directory. It is decrypted to a tensor only at the time of inference. A simple custom designed model was trained on a medMNIST dataset (breast ultrasound). This was integrated into a flask web app set up to test private and public key generation and the encryption-decryption pipeline. 

At the current stage (9th commit) the user can upload an image to the portal. It will be resized in the pipeline to 28×28. For each distinct upload a new public and private key would be generated and stored in the uploads directory. These keys will used for encryption and decryption. 


