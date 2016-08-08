require("deldir")
require("ROCR")

set.seed(4)

### number of clusters to visualize
k = 30

### how much should the target variable be used to adjust the clustering?
target.gain = 10

### manufactures spiral distributed data
spiral = function(n, phase=0, from=1, to=6*pi, winding=1, sd=1, r.offset=0) {
    theta = seq(from, to, length.out=n) + phase
    r = theta * winding + r.offset * 2 * pi
    data.frame(x=r*cos(theta)+rnorm(n,0,sd), y=r*sin(theta)+rnorm(n,0,sd))
}

### generate data set
generate = function(n = 2000) {
    red.data = spiral(n/2, to=4*pi)
    red.data$target = 0

    green.data = spiral(n/2, r.offset=1/2, to=4*pi)
    green.data$target = 1

    rbind(red.data, green.data)
}


### computes cluster features along with an estimate of quality
features = function(all.data, k=100, target.gain=5) {
    all.data$z = all.data$target * target.gain
    c.1 = kmeans(as.matrix(all.data[,c("x", "y", "z")]), centers=k, nstart=20)
    c.2 = kmeans(as.matrix(all.data[,c("x", "y")]), centers=c.1$centers[,c("x","y")], iter.max=1)
    c.2$centers = c.1$centers[,c(1,2)]
    centroids = c.1$centers
    vtess = deldir(centroids[,1], centroids[,2])
    confusion = table(c.2$cluster, all.data$target)
    bias = apply(confusion, 1, function(row) {max(row)/sum(row)})
    accuracy = sum(bias * rowSums(confusion)) / sum(confusion)

    list(clusters=c.2, vtess=vtess, confusion=confusion, bias=bias, accuracy=accuracy)
}

assign.clusters = function(data, clusters) {
    c.2 = kmeans(as.matrix(data[,c("x","y")]), centers=clusters$centers, iter.max=1)
    factor(c.2$cluster)
}

### handy pastel colors for making pretty graphs
gray = rgb(0,0,0,alpha=0.2)
red = rgb(1,0,0,alpha=0.3)
green = rgb(0,1,0,alpha=0.3)
transparent = rgb(0,0,0,alpha=0)

### now generate the training data

training.data = generate(2000)

### and some test data
test.data = generate(2000)

### get some cluster features to play with
r = features(training.data, k, target.gain=target.gain)
r.0 = features(training.data, k, target.gain=0)

### show the raw features
pdf(file="fig1.pdf", width=6, height=6, pointsize=14)

par(mar=c(1,1,3,1))
plot(y ~ x, training.data, pch=21, bg=c(green,red)[training.data$target+1], col=transparent, cex=0.6,
     xaxt='n', yaxt='n', xlab=NA, ylab=NA, main=paste("Features with", k, "clusters"))

plot(r.0$vtess, wlines="tess", wpoints="both", number=FALSE, add=T, 
     lty=1, col=gray, cex=0.5, pch=3, bg='black')

dev.off()

### and plot the quality of a classifier running on test data with and without the clusters

### model with cluster features
training.data$cluster = assign.clusters(data=test.data, clusters=r$clusters)
training.data$cluster.hintless = assign.clusters(data=training.data, clusters=r.0$clusters)

m.with = glm(target ~ x + y + cluster, training.data, family='binomial')
m.both = glm(target ~ x + y + cluster + cluster.hintless, training.data, family='binomial')

### model without
m.without = glm(target ~ x + y, training.data, family='binomial')

### finally model with clusters, but no target hinting
m.no.hint = glm(target ~ x + y + cluster.hintless, training.data, family='binomial')

pdf(file="fig2.pdf", width=6, height=6, pointsize=14)


### and show the AUC with hinted clusters
test.data$cluster = assign.clusters(data=test.data, clusters=r$clusters)
test.data$cluster.hintless = assign.clusters(data=test.data, clusters=r.0$clusters)

pred = prediction(predict(m.with, newdata=test.data), training.data$target)
pref = performance(pred, "tpr", "fpr")
plot(pref, main="ROC With and Without Cluster Features")
print(unlist(slot(performance(pred, "auc"), "y.values")))

pred = prediction(predict(m.no.hint, newdata=test.data), training.data$target)
pref = performance(pred, "tpr", "fpr")
plot(pref, add=T, col='blue', lty=2)
print(unlist(slot(performance(pred, "auc"), "y.values")))

if (T) {
    ## and show the AUC with both hinted and unhinted clusters
    ## for large-ish number of clusters, hinting has little effect and using both
    ## may just trigger over-fitting. YMMV as usual
    pred = prediction(predict(m.without, newdata=test.data), training.data$target)
    pref = performance(pred, "tpr", "fpr")
    plot(pref, add=T, col='red')
    print(unlist(slot(performance(pred, "auc"), "y.values")))
}

text(0.6, 0.5, "Without clusters", adj=0)
text(0.1, 0.95, "With clusters", adj=0)
text(0.1, 0.87, "(hinted)", adj=0)
text(0.43, 0.8, "With clusters", adj=0)
text(0.43, 0.73, "(no hints)", adj=0)

abline(0,1,lty=2,col=gray)
dev.off()
