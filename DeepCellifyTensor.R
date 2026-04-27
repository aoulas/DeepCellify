library(reticulate)
use_virtualenv("r-keras3", required = TRUE)
library(keras3)
tf <- import("tensorflow")

library(EBImage)
library(stringr)
library(pbapply)

library(dplyr)
library(ggplot2)
library(plotly)
library(pROC)
library(tidymodels)
library(caret)
library(reshape2)

dataset="COVID" # Choose betwen covif and MM dataset
if(dataset=="COVID"){
  dirwithimages<-"COVID2ModBig" 
  class0<-"Covid"
  class1<-"Healthy"#|LPS|Non.covid
  
}else{
  dirwithimages<-"MM2ModBig" 
  class0<-"MGUS|SMM"
  class1<-"NDMM" #MM|
}
dirwithtrain<-"/mnt/data/oulas/CNNs/Train2/"
dirwithtest<-"/mnt/data/oulas/CNNs/Test2/"
setwd("/mnt/data/oulas/")
dir_path<-paste("/mnt/data/oulas/",dirwithimages,"/data",sep="")


extract_feature <- function(dir_path, width, height, labelsExist = T) {
  img_size <- width * height
  
  ## List images in path
  images_names <- list.files(dir_path)
  images_names<-images_names[grepl("png|jpg|tiff",images_names)]

  images_names<-images_names[grepl(paste("_",class0,"|_",class1,sep=""),images_names)]
  
  if(labelsExist){
    ## Select only class0 or class1 images
    class0_class1 <- str_extract(images_names, paste("(",class0,"|",class1,")",sep=""))
    
    toks=strsplit(class0,"|",fixed = T)
    key<-rep(0, length(toks[[1]]))
    key<-c(key,1)
    keynames<-unlist(toks)
    names(key)<-c(keynames,class1)
    
    y <- key[class0_class1]
  }
  
  print(paste("Start processing", length(images_names), "images"))
  train_array=array(data=NA,dim=c(length(images_names),width, height,  4))
  
  t=0
  for(imgname in images_names){
    
    t=t+1
    if(t%%100==0)message(t)
    result <- tryCatch({
      img <- readImage(file.path(dir_path, imgname))
    }, error = function(e) {
      if (grepl("file is not in PNG format", e$message)) {
        message(paste("Skipping file (not a valid PNG):"))
      } else {
        stop(e)  # Re-throw unexpected errors
      }
    })
    
    img_resized <- resize(img, w = width, h = height)
    img_matrix <- img_resized@.Data
    train_array[t,,,]=img_matrix
  }
  if(labelsExist){
    return(list(X = train_array, y = y))
  }else{
    return(train_array)
  }
}

file.remove(list.files(dirwithtest,full.names = T))
file.remove(list.files(dirwithtrain,full.names = T))


width <- 30
height <- 30

images_names <- list.files(dir_path)
images_names<-gsub("^_","",images_names)

images_names<-images_names[grepl(paste("_",class0,"|_",class1,sep=""),images_names)]
images_names<-images_names[grepl("png",images_names)]

if(dataset=="COVID"){
  #For COVID_H###########################################################################################################################################################
  SamplNames <- unique(str_extract(images_names, "_.+[0-9]+[s]?\\_"))
  SamplNamesLabels <- unique(str_extract(images_names, paste("_.+[0-9]+[s]?\\_","(",class0,"|",class1,")",sep="")))

  SamplNames<-gsub("[_\\.]","",SamplNames)
  SamplNames<-SamplNames[!is.na(SamplNames)]
  names(SamplNamesLabels)<-SamplNames
  SamplNames<-sort(SamplNames)

  unique(SamplNames)
  ##########################################################################################################################################################
}else{
  #For MM data####################################################################################################################################
  SamplNames <- unique(str_extract(images_names, "_.+[0-9]+[s]?\\."))
  SamplNames<-gsub("[_\\.]","",SamplNames)
  SamplNames<-SamplNames[!is.na(SamplNames)]
  SamplNames<-sort(SamplNames)
  unique(SamplNames)
  ##################################################################################################################################################
}
CellTypes<-unique(gsub("_.+","",images_names))

tableofres<-c()
#run_index<-1
runcomb<-FALSE
if(runcomb){
  CellTypes<-apply(combn(CellTypes,2),2,paste,collapse='|')
}


for(celltype in CellTypes[12]){
  aucs<-c()
  plot_list <- list()  # initialize empty list
  i_plot<-1
  for(samp in SamplNames){
    images_names <- list.files(dir_path)
    if(runcomb){
      toks<-strsplit(celltype,"\\|")
      images_names1<-images_names[grepl(paste("_",toks[[1]][1],"_",samp,sep=""),images_names)]
      images_names2<-images_names[grepl(paste("_",toks[[1]][2],"_",samp,sep=""),images_names)]
      images_names<-c(images_names1,images_names2)
    }else{
      if(dataset=="COVID"){
        ##################for COVID#############################################################
        images_names<-images_names[grepl(paste("_",celltype,"_",samp,"_",sep=""),images_names)]
        ########################################################################################
      }else{
        images_names<-images_names[grepl(paste("_",celltype,".+",samp,sep=""),images_names)]
      }
    }
    
    file.copy(paste("/mnt/data/oulas/",dirwithimages,"/data/",images_names,sep=""), dirwithtest)
    
    images_names <- list.files(dir_path)
    if(runcomb){
      images_names1<-images_names[grepl(paste("_",toks[[1]][1],sep=""),images_names)]
      images_names2<-images_names[grepl(paste("_",toks[[1]][2],sep=""),images_names)]
      images_names1<-images_names1[!grepl(paste("_",toks[[1]][1],"_",samp,sep=""),images_names1,)]
      images_names2<-images_names2[!grepl(paste("_",toks[[1]][2],"_",samp,sep=""),images_names2,)]
      images_names<-c(images_names1,images_names2)
    }else{
      if(dataset=="COVID"){
        ##################for COVID#############################################################
        images_names<-images_names[grepl(paste("_",celltype,"_",sep=""),images_names)]
        images_names<-images_names[!grepl(paste("_",celltype,"_",samp,sep=""),images_names,)]
        ########################################################################################
      }else{
        images_names<-images_names[grepl(paste("_",celltype,sep=""),images_names)]
        images_names<-images_names[!grepl(paste("_",celltype,".+",samp,sep=""),images_names,)]
      }
    }
    
    
    file.copy(paste("/mnt/data/oulas/",dirwithimages,"/data/",images_names,sep=""), dirwithtrain)
    Sys.sleep(5)
    train=extract_feature(dirwithtrain, width, height)
    train_array=train$X
    test=extract_feature(dirwithtest, width, height, labelsExist = F)
    
    # Check processing on first sample
    # par(mar = rep(0, 4))
    # testsamp <- t(matrix(as.numeric(train_array[1,,,]),
    #                     nrow = width, ncol = height, T))
    # image(t(apply(testsamp, 1, rev)), col = gray.colors(12),
    #       axes = F)
    
    
    
    model <- keras_model_sequential() 
    inputs <- layer_input(shape = c(width, height, 4))
    
    x <- inputs %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", padding = "same") %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", padding = "valid") %>%
      layer_max_pooling_2d(pool_size = 2) %>%
      layer_dropout(rate = 0.25) %>%
      
      layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", padding = "same", strides = 2) %>%
      layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", padding = "valid") %>%
      layer_max_pooling_2d(pool_size = 2) %>%
      layer_dropout(rate = 0.25) %>%
      
      layer_flatten() %>%
      layer_dense(units = 50, activation = "relu") %>%
      layer_dropout(rate = 0.25)
    
    outputs <- layer_dense(x, units = 1, activation = "sigmoid")
    
    model <- keras_model(inputs = inputs, outputs = outputs)
    summary(model)
    
    
    specificity_metric <- custom_metric("specificity", function(y_true, y_pred) {
      
      y_true <- tf$cast(y_true, tf$float32)
      y_pred <- tf$cast(tf$round(y_pred), tf$float32)
      
      tn <- tf$reduce_sum(tf$cast((y_true == 0 & y_pred == 0), tf$float32))
      fp <- tf$reduce_sum(tf$cast((y_true == 0 & y_pred == 1), tf$float32))
      
      tn / (tn + fp + tf$keras$backend$epsilon())
    })
    
    model %>% compile(
      loss = 'binary_crossentropy',
      optimizer = "adam",
      metrics = c(metric_auc(name = "auc"),metric_recall(name = "recall"),specificity_metric)#tf$keras$metrics$AUC()#c('accuracy')#metric_precision(),metric_recall()
    )
    
    #Shuffles the data
    x_rand <- sample(1:length(train$y))       
    train$y<-train$y[x_rand]
    train_array<-train_array[x_rand,,,]
    
    
    
    trainIndex <- createDataPartition(as.factor(train$y), p = 0.8, #p = 0.5
                                      list = FALSE, 
                                      times = 1)
    
    
    train_array<-train_array[trainIndex,,,]
    train$y<-train$y[trainIndex]
    
    val_data<-train_array[-trainIndex,,,]
    val_data_labs<-train$y[-trainIndex]
    
    
    countsLabs<-table(train$y)
    countsLabsval<-table(val_data_labs)
    weight<-max(countsLabs)/min(countsLabs)
    higherclass<-names(countsLabs[(countsLabs==max(countsLabs))])
    
    
    LossTerminate <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if ((logs$val_loss / logs$loss) > 1) {
          model$stop_training <- TRUE
        }
      }
    )
    
    early_stop_auc <- callback_early_stopping(
      monitor = "val_auc",
      patience = 5,
      mode = "max",
      min_delta = 0.01,
      restore_best_weights = TRUE
    )
    
    stop_on_auc <- callback_lambda(
      on_epoch_end = function(epoch, logs = NULL) {
        val_auc <- logs[["val_auc"]]
        
        if (!is.null(val_auc) && val_auc >= 0.99) {
          cat(sprintf("\nReached target val_auc: %.4f — stopping training\n", val_auc))
          model$stop_training <- TRUE
        }
      }
    )
    
    Terminate <- LossTerminate#$new()
    epoch_val<-50
    if(length(higherclass) > 1){
      history <- model %>% fit(
        x = train_array, y = as.numeric(train$y), 
        epochs = epoch_val, batch_size = 50,shuffle=FALSE, callbacks = list(Terminate,early_stop_auc,stop_on_auc),
        #validation_split = 0.2 
        validation_split = 0.5,validation_data = list(val_data,val_data_labs)
      )
    }else{
      if(higherclass=="1"){
        history <- model %>% fit(
          x = train_array, y = as.numeric(train$y), 
          epochs = epoch_val, batch_size = 50,shuffle=FALSE, callbacks = list(Terminate,early_stop_auc,stop_on_auc),
          #validation_split = 0.2,
          validation_split = 0.5,validation_data = list(val_data,val_data_labs),class_weight = list("0"=weight,"1"=1)
        )
      }else{
        history <- model %>% fit(
          x = train_array, y = as.numeric(train$y), 
          epochs = epoch_val, batch_size = 50,shuffle=FALSE, callbacks = list(Terminate,early_stop_auc,stop_on_auc),
          #validation_split = 0.2,
          validation_split = 0.5,validation_data = list(val_data,val_data_labs),class_weight = list("0"=1,"1"=weight)
        )
      }
    }
    p_his<-plot(history)
    write.table(history, file = paste("histories/",samp,"_history.txt",sep=""), sep = "\t", row.names = FALSE, col.names = FALSE)
    print(p_his)
    plot_list[[i_plot]] <- p_his  # save plot to list
    i_plot<-i_plot+1
    pdb<-as.data.frame(history$metrics)
    
    pred_probs <- model %>% predict(val_data)
    predictions <- ifelse(pred_probs > 0.5, 1, 0)
    
    probabilities <- model %>% predict(val_data)
    
    valdata<-as.data.frame(cbind(val_data_labs,round(probabilities,5)))
    
    colnames(valdata)<-c("Label","Probabilities")
    if(length(unique(valdata$Label))==1){
      if(unique(valdata$Label)==0){
        valdata$Label[1]<-1
      }else{
        valdata$Label[1]<-0
      }
    }
    valdata$Label<-as.factor(valdata$Label)
    
    
    pdbval <- roc_curve(valdata, Label, Probabilities,event_level = "second")
    pdbval$specificity <- 1 - pdbval$specificity
    auc = roc_auc(valdata, Label, Probabilities,event_level = "second")
    auc = auc$.estimate
    aucs=c(aucs,auc)
    tit = paste('Validation ROC Curve (AUC = ',toString(round(auc,2)),')',sep = '')
    
    
    fig <-  plot_ly(data = pdbval ,x =  ~specificity, y = ~sensitivity, type = 'scatter', mode = 'lines', fill = 'tozeroy') %>%
      layout(title = tit,xaxis = list(title = "1-Specificity"), yaxis = list(title = "Sensitivity")) %>%
      add_segments(x = 0, xend = 1, y = 0, yend = 1, line = list(dash = "dash", color = 'black'),inherit = FALSE, showlegend = FALSE)
    print(fig)
    
    
    if(length(test)!=0){
      
      pred_probs_test <- model %>% predict(test)
      predictions_test <- ifelse(pred_probs_test > 0.5, 1, 0)
      probabilities <- model %>% predict(test)
      set.seed(101)
      if(nrow(test) <32){
        sampstoshow<-nrow(test)
      }else{
        sampstoshow<-32
      }
      random <- sample(1:nrow(test), sampstoshow)
      par(mfrow = c(4, 8), mar = rep(0, 4),bg='transparent')
      
      dir_pathTest=dirwithtest
      images_names <- list.files(dir_pathTest)
      images_names<-images_names[grepl("png|jpg|tiff",images_names)]
      for(r in random){
        imgname=images_names[r]
        img <- readImage(file.path(dir_pathTest, imgname))
        image(img[,,2],col = gray.colors(12), axes = F)
        
        legend("topright", legend = ifelse(predictions[r,] == 0, class0, class1),
               text.col = ifelse(predictions[r,] == 0, 2, 4), bty = "n", text.font = 2)
        legend("topleft", legend = as.vector(round(probabilities[r,], 2)), bty = "n", col = "white")
        legend("center", legend = imgname, bty = "n", col = "white")
      }
      
      
      if(class0%in%samp){
        testdata<-as.data.frame(cbind(rep(0, nrow(probabilities)),probabilities))
        testdata[1,1]<-1
      }else{
        testdata<-as.data.frame(cbind(rep(1, nrow(probabilities)),probabilities))
        testdata[1,1]<-0
      }
      
      colnames(testdata)<-c("Label","Probabilities")
      testdata$Label<-as.factor(testdata$Label)
      if(nrow(testdata) > 1){
        pdbtest <- roc_curve(testdata, Label, Probabilities,event_level = "second")
        pdbtest$specificity <- 1 - pdbtest$specificity
        auc = roc_auc(testdata, Label, Probabilities,event_level = "second")
        auc = auc$.estimate
        
        tit = paste('Test ROC Curve (AUC = ',toString(round(auc,2)),')',sep = '')
        
        
        fig <-  plot_ly(data = pdbtest ,x =  ~specificity, y = ~sensitivity, type = 'scatter', mode = 'lines', fill = 'tozeroy') %>%
          layout(title = tit,xaxis = list(title = "1-Specificity"), yaxis = list(title = "Sensitivity")) %>%
          add_segments(x = 0, xend = 1, y = 0, yend = 1, line = list(dash = "dash", color = 'black'),inherit = FALSE, showlegend = FALSE)
        print(fig)
        
      }
      
      celllabels<-ifelse(predictions == 0, class0, class1)
      if(dataset=="COVID"){
        #for COVID_H##########################################################################
        samplpluslabel<-gsub("^_","",SamplNamesLabels[which(names(SamplNamesLabels)==samp)])
        #print(paste(samp,samplpluslabel))
        #####################################################################################
      }
      print(samp)
      print(table(celllabels))
      
      tableofressamp<-t(table(celllabels))
      #check is all test data are classifies as one class
      if(length(unique(celllabels))==1){
        if(unique(celllabels)==class0){
          tableofressamp<-cbind(tableofressamp,0)
        }else{
          tableofressamp<-cbind(0,tableofressamp)
        }
        colnames(tableofressamp)<-c(class0,class1)
      }
      
      if(dataset=="COVID"){
        #for COVID_H##########################################################################
        rownames(tableofressamp)<-paste(samplpluslabel,"_",celltype,sep="")
        #####################################################################################
      }else{
        rownames(tableofressamp)<-paste(samp,"_",celltype,sep="")
      }
      
      tableofres<-rbind(tableofres,tableofressamp)
    }else{
      tableofressamp<-t(as.table(c(0,0)))
      
      if(dataset=="COVID"){
        #for COVID_H##########################################################################
        samplpluslabel<-gsub("^_","",SamplNamesLabels[which(names(SamplNamesLabels)==samp)])
        rownames(tableofressamp)<-paste(samplpluslabel,"_",celltype,sep="")
        #####################################################################################
      }else{
        rownames(tableofressamp)<-paste(samp,"_",celltype,sep="")
      }
      
      colnames(tableofressamp)<-c(class0,class1)
      tableofres<-rbind(tableofres,tableofressamp)
      
    }
    file.remove(list.files(dirwithtest,full.names = T))
    file.remove(list.files(dirwithtrain,full.names = T))
    gc()
  }
}

