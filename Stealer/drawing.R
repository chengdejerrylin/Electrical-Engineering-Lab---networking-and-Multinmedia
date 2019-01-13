library(ggplot2)
library(reshape2)
raw <- read.csv("./MNIST_BigMLStealer_3240_result.csv")
head(raw)
raw_deepnet <- read.csv("./MNIST_deepnet_BigMLStealer_3240_result.csv")
head(raw_deepnet)
raw.epoch.200 <- raw[raw$epoch == 200,]
raw.epoch.200.deepnet <- raw_deepnet[raw_deepnet$epoch == 200,]
head(raw.epoch.200)

## Whole data for training
draw.melt.training <- melt(raw.epoch.200, id=c("training.size", "testing.size", "loss.function", "batch.size", "learning.rate", "epoch", "control.testing.accuracy", "copy.testing.accuracy"))
head(draw.melt.training)

## Whole data for testing
draw.melt.testing <- melt(raw.epoch.200, id=c("training.size", "testing.size", "loss.function", "batch.size", "learning.rate", "epoch", "control.trainning.accuracy", "copy.trainning.accuracy"))
head(draw.melt.testing)

## Training with different learning rate
draw.melt.training.learningrate.1 <- draw.melt.training[draw.melt.training$learning.rate == 1e-05, ]
draw.melt.training.learningrate.2 <- draw.melt.training[draw.melt.training$learning.rate == 5e-05, ]
draw.melt.training.learningrate.3 <- draw.melt.training[draw.melt.training$learning.rate == 1e-04, ]

## Testing with different learning rate
draw.melt.testing.learningrate.1 <- draw.melt.testing[draw.melt.testing$learning.rate == 1e-05, ]
draw.melt.testing.learningrate.2 <- draw.melt.testing[draw.melt.testing$learning.rate == 5e-05, ]
draw.melt.testing.learningrate.3 <- draw.melt.testing[draw.melt.testing$learning.rate == 1e-04, ]

# Drawing for whole training data
ggplot(draw.melt.training, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Whole Training Data") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top") +
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Training Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/whole_training.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for testing data
ggplot(draw.melt.testing, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Whole Testing Data") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Testing Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/whole_testing.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 1e-05 
ggplot(draw.melt.training.learningrate.1, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Training + learning rate 1e-05") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Training Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/training_learning_1e-05.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 5e-05 
ggplot(draw.melt.training.learningrate.2, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Training + learning rate 5e-05") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Training Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/training_learning_5e-05.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 1e-04
ggplot(draw.melt.training.learningrate.3, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Training + learning rate 1e-04") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Training Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/training_learning_1e-04.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 1e-05 
ggplot(draw.melt.testing.learningrate.1, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Testing + learning rate 1e-05") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Testing Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/testing_learning_1e-05.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 5e-05
ggplot(draw.melt.testing.learningrate.2, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Testing + learning rate 5e-05") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Testing Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/testing_learning_5e-05.png",
       dpi = 300,
       width = 7,
       height = 7)

# Drawing for whole training data, learning rate 1e-04
ggplot(draw.melt.testing.learningrate.3, aes(x=training.size, y=value, color=variable)) + 
  ggtitle("Testing + learning rate 1e-04") +
  geom_point(size = 0.8) + 
  theme_bw() +
  theme(legend.position="top")+
  theme(legend.background = element_rect(size=0.5, linetype="solid", 
                                         colour ="darkblue")) + 
  xlab("Training size") + ylab("Testing Accuracy") + 
  theme(plot.title = element_text(size=25, hjust = 0.5))
ggsave("figure/testing_learning_1e-04.png",
       dpi = 300,
       width = 7,
       height = 7)
