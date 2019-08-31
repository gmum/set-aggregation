import numpy as np
import tensorflow as tf
import tqdm as tqdm
from src.metrics.metrics import accurracy


class Model:
    def __init__(self, DataProvider, BodyBuilder,
                 HeadBuilder, PlaceholderBuilder, Monitor, scopes,
                 trainable_scopes, is_training,
                 learning_rate=1e-3, lr_scheduler=None):
        self.DataProvider = DataProvider
        self.BodyBuilder = BodyBuilder
        self.HeadBuilder = HeadBuilder
        self.Monitor = Monitor
        self.scopes = scopes
        self.trainable_scopes = trainable_scopes
        self.init_learning_rate = learning_rate
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.placeholder_builder = PlaceholderBuilder
        self.initialized = False
        self.is_training = is_training
        self.lr_scheduler = lr_scheduler

    def set_up(self):
        self.__set_up_placeholders()
        self.__set_up_model()
        self.__set_up_training()
        self.initialized = True

    def __set_up_placeholders(self):
        self.x, self.y, self.priorities, self.weights = self.placeholder_builder.set_up_placeholders()

    def __set_up_model(self):
        self.processed = self.BodyBuilder.get_body(self.x, self.scopes["body"], self.priorities,
                                                   self.weights)
        self.model_loss, self.pred_logits = self.HeadBuilder.get_head(self.processed, self.y, self.scopes["head"])
        self.pred = tf.nn.softmax(self.pred_logits)
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.model_loss + self.reg_loss

    def __set_up_training(self):
        self.vars = []
        for scope in self.trainable_scopes:
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.vars += trainable_vars
        print(len(self.vars), "TOTAL NUMBER OF PARAMETERS: ",
              np.sum([np.prod(v.get_shape().as_list()) for v in self.vars]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.vars)

    def add_to_monitor_loss(self, loss, train_acc, test_loss, test_acc):
        self.Monitor.monitor_all(["train_loss", "train_acc", "valid_loss", "valid_acc"],
                                 [loss, train_acc, test_loss, test_acc])

    def add_to_monitor(self, loss, pred, test_acc, test_loss, test_pred, train_acc):
        self.add_to_monitor_loss(loss,train_acc,test_loss,test_acc)
        self.Monitor.monitor_all(["train_pred", "test_pred"],
                                 [np.argmax(pred, axis=1), np.argmax(test_pred, axis=1)])

    def train(self, session, epochs, batch_size, args, saver, config_name, model_name):
        assert self.initialized, "model must be set up before training"
        self.Monitor.save_args(args, config_name)
        self.DataProvider.load_dataset(args.shuffle, args.random_state)
        for epoch in range(epochs):
            for iter in tqdm.tqdm(range(len(self.DataProvider) // batch_size)):
                train_x, train_y, weights, priorities = self.DataProvider.get_random_batch(batch_size, "train")
                lr = self.lr_scheduler(epoch) if self.lr_scheduler is not None else self.init_learning_rate
                if self.weights is not None and self.priorities is not None:
                    _, loss, pred = session.run([self.train_op, self.loss, self.pred],
                                                feed_dict={self.x: train_x, self.y: train_y, self.weights: weights,
                                                           self.priorities: priorities, self.is_training:True,
                                                           self.learning_rate:lr})
                else:
                    _, loss, pred = session.run([self.train_op, self.loss, self.pred],
                                                feed_dict={self.x: train_x, self.y: train_y, self.is_training:True,
                                                           self.learning_rate: lr
                                                           })

                test_x, test_y, test_weights, test_priorities = self.DataProvider.get_random_batch(batch_size, "valid")

                if self.weights is not None and self.priorities is not None:
                    test_loss, test_pred = session.run([self.loss, self.pred],
                                                       feed_dict={self.x: test_x, self.y: test_y,
                                                                  self.weights: test_weights,
                                                                  self.priorities: test_priorities,
                                                                  self.is_training: False})
                else:
                    test_loss, test_pred = session.run([self.loss, self.pred],
                                                       feed_dict={self.x: test_x,
                                                                  self.y: test_y,
                                                                  self.is_training: False})
                train_acc = accurracy(np.argmax(pred, axis=1), train_y)
                test_acc = accurracy(np.argmax(test_pred, axis=1), test_y)
                self.add_to_monitor(loss, pred, test_acc, test_loss, test_pred, train_acc)
            self.Monitor.save_session(session, saver, model_name)
            self.Monitor.save()

    def train_epoch(self, session, epochs, batch_size, args, saver, config_name, model_name):
        assert self.initialized, "model must be set up before training"
        self.Monitor.save_args(args, config_name)
        self.DataProvider.load_dataset(args.shuffle, args.random_state)
        for epoch in range(epochs):
            lr = self.lr_scheduler(epoch) if self.lr_scheduler is not None else self.init_learning_rate
            Loss, Acc = [], []
            for iter in tqdm.tqdm(range(len(self.DataProvider) // batch_size)):
                train_x, train_y, weights, priorities = self.DataProvider.get_next_batch(batch_size, iter, "train")

                if self.weights is not None and self.priorities is not None:
                    _, loss, pred = session.run([self.train_op, self.loss, self.pred],
                                                feed_dict={self.x: train_x, self.y: train_y, self.weights: weights,
                                                           self.priorities: priorities, self.is_training:True,
                                                           self.learning_rate:lr})
                else:
                    _, loss, pred = session.run([self.train_op, self.loss, self.pred],
                                                feed_dict={self.x: train_x, self.y: train_y, self.is_training:True,
                                                           self.learning_rate: lr})

                train_acc = accurracy(np.argmax(pred, axis=1), train_y)
                self.Monitor.monitor_all(["train_pred"], [np.argmax(pred, axis=1)])
                Loss.append(loss)
                Acc.append(train_acc)

            Loss_valid, Acc_valid = [], []
            for v_iter in range(self.DataProvider.valid_len() // batch_size):
                test_x, test_y, test_weights, test_priorities = self.DataProvider.get_next_batch(batch_size, v_iter, "valid")
                
                if self.weights is not None and self.priorities is not None:
                    test_loss, test_pred = session.run([self.loss, self.pred],
                                                       feed_dict={self.x: test_x, self.y: test_y,
                                                                  self.weights: test_weights,
                                                                  self.priorities: test_priorities,
                                                                  self.is_training:False})
                else:
                    test_loss, test_pred = session.run([self.loss, self.pred],
                                                       feed_dict={self.x: test_x, self.y: test_y,
                                                                  self.is_training:False})

                self.Monitor.monitor_all(["valid_pred"], [np.argmax(test_pred, axis=1)])
                test_acc = accurracy(np.argmax(test_pred, axis=1), test_y)
                Loss_valid.append(test_loss)
                Acc_valid.append(test_acc)

            self.add_to_monitor_loss(np.mean(Loss), np.mean(Acc), np.mean(Loss_valid), np.mean(Acc_valid))
            self.Monitor.save_session(session, saver, model_name)
            self.Monitor.save()


    def predict(self, session, X):
        pred = session.run(self.pred, feed_dict={self.x: X, self.is_training: False})
        return np.argmax(pred, axis=1)

    def predict_dataset(self, session, batch_size, dataset_type, args):
        self.DataProvider.load_dataset(args.shuffle, args.random_state)
        length = self.DataProvider.get_dataset_length(dataset_type)
        for iter in range(length // batch_size):
            batch_x, batch_y, ww, pp = self.DataProvider.get_next_batch( batch_size, iter, dataset_type)
            if self.weights is not None and self.priorities is not None:
               loss, pred = session.run([self.loss, self.pred],
                                     feed_dict={self.x: batch_x, self.y: batch_y, self.weights: ww,
                                                self.priorities: pp, self.is_training: False})
            else:
               loss, pred = session.run([self.loss, self.pred],
                                     feed_dict={self.x:batch_x, self.y:batch_y, self.is_training: False})
           
            acc = accurracy(np.argmax(pred, axis=1), batch_y)
            self.Monitor.add_variable("trained_model_" + dataset_type + "_loss", loss)
            self.Monitor.add_variable("trained_model_" + dataset_type + "_accuracy", acc)
        self.Monitor.save()

    def get_init_variables(self):
        return self.vars


class MultiImageModel(Model):
    def __init__(self, DataProvider, BodyBuilder, HeadBuilder, MultiImagePlaceholderBuilder, Monitor, scopes,
                 trainable_scopes, shapes, is_training, learning_rate=1e-3, lr_scheduler=None, train_mode=None):
        super().__init__(DataProvider, BodyBuilder, HeadBuilder, MultiImagePlaceholderBuilder, Monitor, scopes,
                         trainable_scopes,is_training, learning_rate, lr_scheduler)
        self.shapes = shapes
        self.train_mode=train_mode

    def set_up(self):
        self.__set_up_placeholders()
        self.__set_up_model()
        self.__set_up_training()
        self.initialized = True


    def __set_up_placeholders(self):
        self.x, self.y, self.priorities, self.weights = self.placeholder_builder.set_up_placeholders()

    def __set_up_model(self):
        model_loss, pred_logits, processed, pred, reg_loss, loss = self.__get_model_lists()
        for i, x in enumerate(self.x):
            processed[i] =self.BodyBuilder.get_body(x, self.scopes["body"], self.priorities[i],
                                                       self.weights[i])
            model_loss[i], pred_logits[i] = self.HeadBuilder.get_head(processed[i], self.y[i], self.scopes["head"])
            pred[i] = tf.nn.softmax(pred_logits[i])
            reg_loss[i] = tf.losses.get_regularization_loss()
            loss[i] = model_loss[i] + reg_loss[i]
        self.model_loss = model_loss
        self.pred_logits = pred_logits
        self.processed = processed
        self.pred = pred
        self.reg_loss = reg_loss
        self.losses = loss
        self.sum_loss = tf.reduce_sum(self.losses)


    def __get_model_lists(self):
        processed = [None] * len(self.shapes)
        model_loss = [None] * len(self.shapes)
        pred_logits = [None] * len(self.shapes)
        pred = [None] * len(self.shapes)
        reg_loss = [None] * len(self.shapes)
        loss = [None] * len(self.shapes)
        return model_loss, pred_logits, processed, pred, reg_loss, loss

    def __set_up_training(self):
        self.vars = []
        for scope in self.trainable_scopes:
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.vars += trainable_vars
        print(len(self.vars), "TOTAL NUMBER OF PARAMETERS: ",
              np.sum([np.prod(v.get_shape().as_list()) for v in self.vars]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        if self.train_mode is None or len(self.train_mode)==0:
            self.train_op = self.optimizer.minimize(self.sum_loss, var_list=self.vars)
            self.loss = self.sum_loss
        else:
            self.total_loss = self.losses[self.train_mode[0]]
            for i in self.train_mode[1:]:
                self.total_loss += self.losses[i]
            self.train_op = self.optimizer.minimize(self.total_loss, var_list=self.vars)
            self.loss = self.total_loss

    def add_to_monitor_loss(self, loss, train_acc, test_loss, test_acc):
        self.Monitor.monitor_all(["train_loss", "train_acc", "valid_loss", "valid_acc"],
                                 [loss, train_acc, test_loss, test_acc])

    def add_to_monitor_loss_i(self, loss, train_acc, test_loss, test_acc, i):
        self.Monitor.monitor_all(["train_loss_{}".format(i), "train_acc_{}".format(i),
                                  "valid_loss_{}".format(i), "valid_acc_{}".format(i)],
                                 [loss, train_acc, test_loss, test_acc])


    def add_to_monitor(self, loss, pred, test_acc, test_loss, test_pred, train_acc):
        self.add_to_monitor_loss(loss, train_acc, test_loss, test_acc)
        self.Monitor.monitor_all(["train_pred", "test_pred"],
                                 [np.argmax(pred, axis=1), np.argmax(test_pred, axis=1)])

    def add_to_monitor_i(self, loss, pred, test_acc, test_loss, test_pred, train_acc, i):
        self.add_to_monitor_loss_i(loss, train_acc, test_loss, test_acc, i)
        self.Monitor.monitor_all(["train_pred_{}".format(i), "test_pred_{}".format(i)],
                                 [np.argmax(pred, axis=1), np.argmax(test_pred, axis=1)])

    def prepare_feed_dict(self, train_x, train_y, weights=None, priorities=None, is_training=True,  epoch=None):
        feed_dict = {self.is_training: is_training}
        if train_y is not None:
            for i, (xx, yy) in enumerate(zip(self.x,self.y)):
                feed_dict[xx]=train_x[i]
                feed_dict[yy]=train_y[i]
                if weights is not None and priorities is not None:
                    if(self.weights[i] is not None and self.priorities[i] is not None):
                        feed_dict[self.weights[i]]=weights[i]
                        feed_dict[self.priorities[i]]=priorities[i]
        else:
            for i, (xx, yy) in enumerate(zip(self.x, self.y)):
                feed_dict[xx]=train_x[i]
        if is_training and epoch is not None:
            lr = self.lr_scheduler(epoch) if self.lr_scheduler is not None else self.init_learning_rate
            feed_dict[self.learning_rate] = lr
        return feed_dict


    def monitor_losses(self, session, train_x, train_y, test_x, test_y, weights,priorities, test_weights, test_priorities):
        for i in range(len(self.shapes)):
            loss_i = session.run(self.loss[i], feed_dict=self.prepare_feed_dict(train_x, train_y, weights, priorities, False))

    def train(self, session, epochs, batch_size, args, saver, config_name, model_name):
        assert self.initialized, "model must be set up before training"
        self.Monitor.save_args(args, config_name)
        self.DataProvider.load_dataset(args.shuffle, args.random_state)
        for epoch in range(epochs):
            for iter in tqdm.tqdm(range(len(self.DataProvider) // batch_size)):
                train_x, train_y, weights, priorities = self.DataProvider.get_random_batch(batch_size, "train")
                _, loss, losses, pred = session.run([self.train_op, self.loss,self.losses, self.pred],
                                                feed_dict=self.prepare_feed_dict(train_x,train_y,weights,priorities,True,epoch))

                test_x, test_y, test_weights, test_priorities = self.DataProvider.get_random_batch(batch_size, "valid")

                test_loss, test_losses, test_pred = session.run([self.loss, self.losses, self.pred],
                                                       feed_dict=self.prepare_feed_dict(test_x, test_y, test_weights,
                                                                                        test_priorities,False))
                train_acc = []
                test_acc= []
                for i in range(len(self.x)):
                    train_acc.append(accurracy(np.argmax(pred[i], axis=1), train_y[i]))
                    test_acc.append(accurracy(np.argmax(test_pred[i], axis=1), test_y[i]))
                    self.add_to_monitor_i(losses[i],pred[i],test_acc[i],test_losses[i],test_pred[i],train_acc[i],i)
                self.add_to_monitor(loss, np.concatenate(pred,axis=0), np.mean(test_acc), test_loss, np.concatenate(test_pred,axis=0), np.mean(train_acc))
            self.Monitor.save_session(session, saver, model_name)
            self.Monitor.save()

    def train_epoch(self, session, epochs, batch_size, args, saver, config_name, model_name):
        assert self.initialized, "model must be set up before training"
        self.Monitor.save_args(args, config_name)
        self.DataProvider.load_dataset(args.shuffle, args.random_state)
        for epoch in range(epochs):
            Loss, Acc = [], []
            for iter in tqdm.tqdm(range(len(self.DataProvider) // batch_size)):
                train_x, train_y, weights, priorities = self.DataProvider.get_next_batch(batch_size, iter, "train")

                _, loss, losses, pred = session.run([self.train_op, self.loss, self.losses, self.pred],
                                                    feed_dict=self.prepare_feed_dict(train_x, train_y, weights,
                                                                                     priorities, True, epoch))

                train_acc = []
                for i in range(len(self.x)):
                    train_acc.append(accurracy(np.argmax(pred, axis=1), train_y))
                    self.Monitor.monitor_all_(["train_pred_{}".format(i)], [np.argmax(pred[i], axis=1)])
                Loss.append(loss)
                Acc.append(np.mean(train_acc))


            Loss_valid, Acc_valid = [], []
            for v_iter in range(self.DataProvider.valid_len() // batch_size):
                test_x, test_y, test_weights, test_priorities = self.DataProvider.get_next_batch(batch_size, v_iter,"valid")


                test_loss, test_losses, test_pred = session.run([self.loss, self.losses, self.pred],
                                                                feed_dict=self.prepare_feed_dict(test_x, test_y,
                                                                                                 test_weights,
                                                                                                 test_priorities, False))
                test_acc = []
                for i in range(len(self.x)):
                    test_acc.append(accurracy(np.argmax(test_pred, axis=1), test_y))
                    self.Monitor.monitor_all_(["valid_pred_{}".format(i)], [np.argmax(test_pred[i], axis=1)])

                Loss_valid.append(test_loss)
                Acc_valid.append(np.mean(test_acc))

            self.add_to_monitor_loss(np.mean(Loss), np.mean(Acc), np.mean(Loss_valid), np.mean(Acc_valid))
            self.Monitor.save_session(session, saver, model_name)
            self.Monitor.save()

    def predict(self, session, X):
        res_pred = []
        if len(X) == len(self.x):
            for i in range(len(self.x)):
               pred = session.run(self.pred[i], feed_dict=self.prepare_feed_dict(X,None,is_training=False))
               res_pred.append(np.argmax(pred, axis=1))
        return np.concatenate(res_pred,axis=0)

    def predict_dataset(self, session, batch_size, dataset_type, args):
        if not self.DataProvider.loaded:
                self.DataProvider.load_dataset(args.shuffle, args.random_state)
        length = self.DataProvider.get_dataset_length(dataset_type)
        for iter in range(length // batch_size):
            batch_x, batch_y, ww, pp = self.DataProvider.get_next_batch(batch_size, iter, dataset_type)

            loss, losses, pred = session.run([self.loss, self.losses, self.pred],
                                                            feed_dict=self.prepare_feed_dict(batch_x, batch_y,
                                                                                             ww, pp, is_training=False))
            acc=[]
            for i in range(len(self.x)):
                acc.append(accurracy(np.argmax(pred[i], axis=1), batch_y[i]))
                self.Monitor.add_variable("trained_mode_" + dataset_type +"_accuracy_{}".format(i),acc[i])
            self.Monitor.add_variable("trained_model_" + dataset_type + "_loss", loss)
            self.Monitor.add_variable("trained_model_" + dataset_type + "_accuracy", np.mean(acc))
        self.Monitor.save()

    def get_init_variables(self):
        return self.vars

