import numpy as np
from sklearn.cluster import KMeans
from configs.esd_config import * 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm


def load_data():
    datadict = np.load(datadict_path, allow_pickle=True).item()
    return datadict

def cluster(datadict):
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(datadict['feat'])
    cluster_labels = kmeans.fit_predict(datadict['feat'])
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(datadict['feat'])
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cluster_labels, cmap='viridis')
    plt.title('Cluster Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    breakpoint()
    plt.savefig(os.path.join(results_path, 'cluster.png'))

def classifier(datadict):
    X_train, X_test, y_train, y_test = train_test_split(datadict['feat'], datadict['class_labels'], test_size=0.2, random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print('accuracy: ', accuracy)
    print('classification report: ', cr)
    breakpoint()

def SVM_cluster(datadict):
    X_train, X_test, y_train, y_test = train_test_split(datadict['feat'], datadict['class_labels'], test_size=0.2, random_state=0)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print('accuracy: ', accuracy)
    print('classification report: ', cr)
    breakpoint()
if __name__ == '__main__':
    datadict = load_data()
    # cluster(datadict)
    # classifier(datadict)
    SVM_cluster(datadict)