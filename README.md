# VQ-VAE via Roundtrip Prior

Esta es una implementación en Pytorch de vector quantized variational autoencoder (https://arxiv.org/abs/1711.00937). 

Se puede encontrar la [implementación original en Tensorflow aquí](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py).

### Correr el modelo VQ-VAE

Para correr el modelo VQ-VAE primero se tiene que importar el archivo `models/vqvae.py`. Luego se puede entrenar el modelo utilizando el método `train`, o se puede cargar el modelo entrenado con CIFAR-10 luego de 200 épocas de la carpeta `checkpoints/vqvae_checkpoint`.

### Roundtrip Prior

Para samplear utilizando como prior el modelo Roundtrip, al igual que en VQ-VAE se puede entrenar el modelo mediante el método `train` con el espacio latente del VQ-VAE, o se puede cargar el modelo ya entrenado de la carpeta checkpoints.

### PixelCNN Prior
Para samplear utilizando como prior el modelo PixelCC, al igual que en VQ-VAE se puede entrenar el modelo mediante el método `train` con el espacio latente del VQ-VAE, o se puede cargar el modelo ya entrenado de la carpeta `checkpoints/pixelcnn`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
