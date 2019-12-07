import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
Name: Huerta, Emilia (Please write names in <Last Name, First Name> format)

Collaborators: Kitamura, Masao (Please write names in <Last Name, First Name> format)

Collaboration details:  Synthesize function
                        Implementation of the error
"""

CELEBA_DIRPATH = 'celebA_dataset'
N_HEIGHT = 78
N_WIDTH = 78
N_TRAIN = 850

def get_eigenfaces(eigenvalues, eigenvectors, k):
  """
    Sorts the eigenvector by eigenvalues
    Returns the projection matrix (eigenfaces)

    faces_centered : N x d vector
      mean subtracted faces
    eigenvalues : 1 x d vector
      eigenvalues
    eigenvectors : d x d vector
      eigenvectors
    k : int
      number of eigenfaces to produce

    returns d x k vector
  """
  #sort eigenvalues
  order = np.argsort(-eigenvalues)
  #take order
  values = eigenvalues[order]
  #sort eigenvectors
  vectors = eigenvectors[:, order]
  #select from o to k
  eigenfaces = vectors[:, 0:k].real
  #eigenvectors
  return eigenfaces

def project(faces, faces_mean, eigenfaces):
  """
    Returns the projected faces (lower dimensionality)

    faces : N x d vector
      images of faces
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x k vector
  """
  #Slide #28 - Compute the mean & center the data (faces)
  #B = x - mean of x
  B = faces - faces_mean

  #Multiply centered faces with eigenfaces
  return np.matmul(B, eigenfaces)  #(B_train, W)

def reconstruct(faces_projected, faces_mean, eigenfaces):
  """
    Returns the reconstructed faces (back-projected)

    faces_projected : N x k vector
      faces in lower dimensions
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x d vector
  """
  #Slide #29 - Recover oour data
  #X_train_hat = np.matmul(Z_train, W.T) + mu_train
  return np.matmul(faces_projected, eigenfaces.T) + faces_mean

def synthesize(eigenfaces, variances, faces_mean, k=50, n=25):
  """
    Synthesizes new faces by sampling from the latent vector distribution

    eigenfaces : d x k vector
      projection matrix
    variances : 1 x d vector
      variances
    faces_mean : 1 x d vector

    returns synthesized faces
  """
  #Example for Digits
  #sample from distribution of Z
  #np.random(0, np.sqrt(sigma))
  Z = np.random.normal(0, np.sqrt(variances[0:k]), (n, variances[0:k].shape[0]))
  X_hat = np.matmul(Z, eigenfaces.T) + faces_mean
  return X_hat



def mean_squared_error(x, x_hat):
  """
    Computes the mean squared error

    x : N x d vector
    x_hat : N x d vector

    returns mean squared error
  """
  return np.mean((x - x_hat) ** 2) #correct


def plot_eigenvalues(eigenvalues):
  """
    Plots the eigenvalues from largest to smallest

    eigenvalues : 1 x d vector
  """
  fig = plt.figure()
  fig.suptitle('Eigenvalues Versus Principle Components')
  plt.plot(eigenvalues)
  plt.show()


def visualize_reconstructions(faces, faces_hat, n=4):
  """
    Creates a plot of 2 rows by n columns
    Top row should show original faces
    Bottom row should show reconstructed faces (faces_hat)

    faces : N x k vector
      images of faces
    faces_hat : 1 x d vector
      images reconstructed faces
  """
  fig = plt.figure()
  fig.suptitle('Real Versus Reconstructed Faces')

  # fig = plt.figure()
  # fig.suptitle('Real Versus Reconstructed Faces: ' + label)

  #Slide #34 - Visualzing the data
  for i in range(0, n * 2):
    ax = fig.add_subplot(2, n, i + 1)
    if i < n:
      ax.imshow(faces[i, ...], cmap='gray')
    else:
      ax.imshow(faces_hat[i - n, ...], cmap='gray')

  plt.show()


def plot_reconstruction_error(mses, k):
  """
    Plots the reconstruction errors

    mses : list
      list of mean squared errors
    ks : list
      list of k used
  """
  fig = plt.figure()
  fig.suptitle('Reconstruction Error')
  plt.xlabel('K values')
  plt.ylabel('MSE')
  plt.plot(k, mses)
  plt.show()


def visualize_eigenfaces(eigenfaces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 eigenfaces (principal components)
    For each dimension k, plot the d number values as an image

    eigenfaces : d x k vector
  """
  #Slide 34 - Visualizing the data
  #faces_train = np.reshape(faces_train, (-1, 78, 78))

  fig = plt.figure()
  fig.suptitle('Top 25 Eigenfaces')

  k = 25
  for i in range(0, k):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(eigenfaces[i, ...], cmap='gray')

  plt.show()

  # fig = plt.figure()
  # fig.suptitle('Top 25 Eigenfaces')
  # for i in range(0, 25):
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   ax.imshow(faces_train[i, ...], cmap='gray')
  # plt.show()

  # fig = plt.figure()
  #fig.suptitle('Top 25 Eigenfaces')
  # for i in range(5 * 5): #25
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   plt.imshow(faces[i, ...], cmap='gray')



    # if i < n:
    #   plt.imshow(faces[i, ...],cmap = 'gray')
    # else:
    #   plt.imshow(reconstructed_faces[i-n, ...], cmap='gray')


def visualize_synthetic_faces(faces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 synthetic faces

    eigenfaces : N x d vector
  """
  #X = ZW^T + U
  #Z = BW projects B to subspace    -------this should give you the projection  np.matmal(B,W)
            #B is training data minus the mean of your training set (centers your data)
            #Passed in a B_train
  #W is eigenfaces which projects
  #wT will project back to original space

  fig = plt.figure()
  fig.suptitle('Synthetic Faces')
  for i in range(0, faces.shape[0]):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(faces[i, ...], cmap='gray')
  plt.show()


if __name__ == '__main__':
  # Load faces from directory
  face_image_paths = glob.glob(os.path.join(CELEBA_DIRPATH, '*.jpg'))

  print('Loading {} images from {}'.format(len(face_image_paths), CELEBA_DIRPATH))
  # Read images as grayscale and resize from (128, 128) to (78, 78)
  faces = []
  for path in face_image_paths:
    im = Image.open(path).convert('L').resize((N_WIDTH, N_HEIGHT))
    faces.append(np.asarray(im))
  faces = np.asarray(faces) # (1000, 78, 78)
  # Normalize faces between 0 and 1
  faces = faces/255.0   #faces is X

  print('Vectorizing faces into N x d matrix')
  # TODO: reshape the faces to into an N x d matrix
  #Slide 26 - reshape the image into a vector
  faces = np.reshape(faces, (-1, 6084))  # 78 * 78 = 6084

  print('Splitting dataset into {} for training and {} for testing'.format(N_TRAIN, faces.shape[0]-N_TRAIN))
  faces_train = faces[0:N_TRAIN, ...]
  faces_test = faces[N_TRAIN::, ...]

  X_train = faces_train
  X_test = faces_test

  print('Computing eigenfaces from training set')
  # TODO: obtain eigenfaces and eigenvalues
  #Slide 28 - compute the mean ans center the data
  mu_train = np.mean(faces_train, axis=0)
  B_train = faces_train - mu_train

  #Compute the covariance matrix
  C = np.matmul(B_train.T, B_train) / (B_train.shape[0])

  # Eigen decomposition
  S, V = np.linalg.eig(C)

  print('Plotting the eigenvalues from largest to smallest')
  # TODO: plot the first 200 eigenvalues from largest to smallest
  # Sort the values in descending order
  #NOT SURE IF THIS WORKS
  top_values = np.sort(S)
  top_values = np.flip(top_values)

  # Select the top 45 dimensions
  #Slide #31

  # top_values = V[0:200]
  k = 200
  top_values = S[0:k]

  plt.title("Top 200 Eigenvalues")
  plt.xlabel('Ranking')
  plt.ylabel('Eigenvalues')
  plt.plot(top_values)
  plt.show()  #takes too long

  print('Visualizing the top 25 eigenfaces')
  # TODO: visualize the top 25 eigenfaces
  # #Slide 34 - Visualizing the data

  # faces_train = np.reshape(faces_train, (-1, 78, 78))
  # fig = plt.figure()
  # fig.suptitle('Top 25 Eigenfaces')
  # for i in range(0, 25):
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   ax.imshow(faces_train[i, ...], cmap='gray')
  # plt.show()

  # fig = plt.figure()
  # for i in range(5 * 5): #25
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   plt.imshow(faces[i, ...], cmap='gray')

  eigenfaces = get_eigenfaces(S, V, k)

  eigenfaces = eigenfaces.T

  eigenfaces = np.reshape(eigenfaces, (-1, 78, 78))

  visualize_eigenfaces(eigenfaces)

  print('Plotting training reconstruction error for various k')
  # TODO: plot the mean squared error of training set with
  k_values=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
  #Got help from Masao
  errors = []

  for k in k_values:
    W = V[:, 0:k]  # (6084, k)
    #Slide #31 - Project our data
    # Project B to Z
    # Z_train = np.matmul(B_train, W)  # (850, k)
    Z_train = project(faces_train, mu_train, W)  # (850, k)

    # Recontruct Z to X_hat
    # X_train_hat = np.matmul(Z_train, W.T) + mu_train  # (850, 6084)
    X_train_hat = reconstruct(Z_train, mu_train, W)  # (850, 6084)

    # Measure loss
    mse = mean_squared_error(X_train, X_train_hat)

    # Append to array
    errors.append(mse)

  plt.title("Mean Squared Error of training set with various K values")
  plt.xlabel('K values')
  plt.ylabel('MSE')
  plt.plot(k_values, errors)
  plt.show()  # Show MSE plot

  print('Reconstructing faces from projected faces in training set')
  # TODO: choose k and reconstruct training set

  # Z is projected faces, and Z = BW
  # W is eigenfaces with dimensions d x k
  # B is centered faces (N x d) X (d x k) = N x k
  # Therefore, Z has fewer dimensions
  # Hence, B was projected to Z

  k = 50
  W = V[:, 0:k].real  # converts complex128 to float64 (needed for imshow to work later)
  # Project B to Z (compress)
  Z_train = np.matmul(B_train, W)  # (850, 50)
  # Reconstruct (decompress)
  X_train_hat = np.matmul(Z_train, W.T) + mu_train
  X_train_hat = np.reshape(X_train_hat, (-1, 78, 78))
  X_train_faces = np.reshape(X_train, (-1, 78, 78))

  # TODO: visualize the reconstructed faces from training set
  visualize_reconstructions(X_train_faces, X_train_hat)

  print('Reconstructing faces from projected faces in testing set')
  # TODO: reconstruct faces from the projected faces
  B_test = X_test - mu_train  # According to Slack chat with Dr. Wong
  C = np.matmul(B_test.T, B_test) / (B_test.shape[0])
  # Project B to Z (compress)
  Z_test = np.matmul(B_test, W)  # (850, 50)
  # Reconstruct (decompress)
  X_test_hat = np.matmul(Z_test, W.T) + mu_train
  X_test_hat = np.reshape(X_test_hat, (-1, 78, 78))
  X_test_faces = np.reshape(X_test, (-1, 78, 78))

  # TODO: visualize the reconstructed faces from training set
  visualize_reconstructions(X_test_faces, X_test_hat)

  print('Plotting testing reconstruction error for various k')
  # TODO: plot the mean squared error of testing set with
  k_values=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

  mses = []
  for k in k_values:
    W = V[:, 0:k]  # (6084, k)

    z_k = np.matmul(B_test, W)  # (150, k)

    x_hat_k = np.matmul(z_k, W.T) + mu_train
    mse = mean_squared_error(X_test, x_hat_k)
    mses.append(mse)

  plot_reconstruction_error(mses, k)

  print('Creating synethic faces')
  # TODO: synthesize and visualize new faces based on the distribution of the latent variables
  # you may choose another k that you find suitable

  k = 50
  eigenfaces = get_eigenfaces(S, V, k)
  variances = S.real
  faces_mean = mu_train
  X_hat = synthesize(eigenfaces, variances, faces_mean)
  X_hat = np.reshape(X_hat, (-1, 78, 78))
  visualize_synthetic_faces(X_hat)
