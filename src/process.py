from sklearn.manifold import TSNE

def visualization(z, pred_labels, total_labels, SNE_n_iter, epoch):   
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=SNE_n_iter)
    tsne_results = tsne.fit_transform(z)
    print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    plt.scatter(tsne_results[1:1000,0],tsne_results[1:1000,1],c=total_labels[1:1000])
    plt.xlabel('latent variable z_1')
    plt.xlabel('latent variable z_2')
    plt.savefig('./z_plot'+str(epoch)+'.png')
    plt.clf()
    
    plt.scatter(tsne_results[1:1000,0],tsne_results[1:1000,1],c=pred_labels[1:1000])
    plt.xlabel('latent variable z_1')
    plt.xlabel('latent variable z_2')
    plt.savefig('./z_classifier_plot'+str(epoch)+'.png')
    plt.clf()
