import numpy as np
import torch
from Args import args
from diff import *
from Log import log_print
from utility import *


class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.n_user = handler.n_user
        self.n_item = handler.n_item
        log_print("USER", self.n_user, "ITEM", self.n_item)
        log_print("NUM OF INTERACTIONS", self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ["Loss", "preLoss", "prediction", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()
        setup_seed(args.random_seed)

    def makePrint(self, name, ep, reses, save):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + "  "
        return ret

    def run(self):
        self.prepareModel()
        log_print("Model Prepared")
        log_print("All Parameters:")
        for arg, value in vars(args).items():
            log_print(f"{arg}: {value}")

        best_epoch = 0
        best_recall_20 = 0
        best_results = ()
        log_print("Model Initialized")
        et = 0
        for ep in range(0, args.epoch):
            if et > 50:
                break
            start_time = time.time()
            tstFlag = ep % args.tstEpoch == 0
            loss = self.trainEpoch()
            if args.report_epoch:
                if ep % 1 == 0:
                    log_print(
                        "Epoch {:03d}; ".format(ep)
                        + "Train loss: {:.4f}; ".format(loss)
                        + "Time cost: "
                        + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time() - start_time)
                        )
                    )
            if tstFlag:
                eval_start = time.time()
                reses = self.testEpoch()
                if reses["Recall"][1] > best_recall_20:
                    best_epoch = ep
                    best_recall_20 = reses["Recall"][1]
                    best_results = reses
                    if not os.path.exists("./result"):
                        os.makedirs("./result")
                    torch.save(
                        self.model.state_dict(),
                        os.path.join("./result", "best_model.pt"),
                    )
                else:
                    et = et + 1
                log_print(
                    "Evalution cost: "
                    + time.strftime("%H: %M: %S", time.gmtime(time.time() - eval_start))
                )
                print_results(None, test_result=reses)
                log_print(
                    "----------------------------------------------------------------"
                )
            # save_best_result(best_results, args.data, args.name)
        print_results(test_result=best_results)

    def prepareModel(self):
        self.topN = [10, 20, 50, 100]
        self.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )
        self.model = Tenc(
            args.hidden_factor,
            self.n_item,
            args.statesize,
            args.dropout_rate,
            args.diffuser_type,
            self.device,
        )
        # args.tstBat = self.n_user
        self.diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
        self.iEmb = nn.Embedding(self.n_item, args.hidden_factor).to(self.device)
        self.uEmb = nn.Embedding(self.n_user, args.hidden_factor).to(self.device)
        nn.init.normal_(self.iEmb.weight, std=0.1)
        nn.init.normal_(self.uEmb.weight, std=0.1)
        social_emb_path = os.path.join(args.data_dir, args.data, "item_social_emb.npy")
        knowledge_emb_path = os.path.join(args.data_dir, args.data, "item_knowledge_emb.npy")
        self.iSoc = torch.from_numpy(np.load(social_emb_path)).float().to(self.device)
        self.iKge = torch.from_numpy(np.load(knowledge_emb_path)).float().to(self.device)

        params_to_optimize = (
            list(self.model.parameters())
            + list(self.iEmb.parameters())
            + list(self.uEmb.parameters())
        )

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        self.model.to(self.device)
        # self.iEmb=torch.tensor(self.handler.iEmb, dtype=torch.float32).to(self.device)
        # self.uEmb=torch.tensor(self.handler.uEmb, dtype=torch.float32).to(self.device)
        # self.tEmb = torch.tensor(self.handler.tEmb, dtype=torch.float32).to(self.device)
        self.graph = self.handler.graph.to(self.device)

    def trainEpoch(self):

        trnLoader = self.handler.trnLoader
        for j, batch in enumerate(trnLoader):
            user, item = batch
            user = user.long().to(self.device)
            pos = item.long().to(self.device)
            negs = trnLoader.dataset.negSampling(pos, user)
            neg = torch.tensor(negs).long().to(self.device)
            self.optimizer.zero_grad()
            
            
            x_start = self.iEmb(pos)
            
            guide_info = self.iSoc[pos]
            n = torch.randint(
                0, args.timesteps, (x_start.shape[0],), device=self.device
            ).long()
            reconloss, predicted_x = self.diff.p_losses(
                self.model, x_start, guide_info, n, loss_type="l2", flag=args.flag
            )
            user_emb0 = self.uEmb(user)
            pos_emb0 = self.iEmb(pos)
            neg_emb0 = self.iEmb(neg)
            user_emb = self.uEmb(user)
            pos_emb = predicted_x
            neg_emb = predicted_x[neg]
            pos_scores = torch.mul(user_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(user_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
            bprloss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            regloss = (
                (1 / 2)
                * (
                    user_emb0.norm(2).pow(2)
                    + pos_emb0.norm(2).pow(2)
                    + neg_emb0.norm(2).pow(2)
                )
                / len(user)
            )
            predicted_x_loss = predicted_x.norm(2).pow(2) / len(item)
            # regloss = user_emb.norm(2).pow(2) / len(user) +\
            #             pos_emb.norm(2).pow(2) / len(item)
            loss = (
                reconloss * (1 - args.bpr_alpha)
                + bprloss * args.bpr_alpha
                + regloss * args.reg_alpha
                + predicted_x_loss
            )
            loss.backward()
            self.optimizer.step()
        return loss

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision, epMRR = None, None, None, None
        num = math.ceil(len(tstLoader.dataset) / args.tstBat)
        for usr, trnMask in tstLoader:
            usr = usr.long().to(self.device)
            trnMask_tensor = trnMask.clone().detach().to(self.device)
            guide_info = self.iSoc
            tst_users_tensor = usr.clone().detach().to(self.device)
            item_emb = self.model.predict(
                self.iEmb.weight, guide_info, self.diff, flag=args.flag
            )
            prediction = compute_prediction(
                self.uEmb.weight,
                item_emb,
                args.n_layers,
                self.graph,
                self.n_user,
                self.n_item,
                tst_users_tensor,
                self.model,
            )
            prediction = prediction * (1 - trnMask_tensor) - trnMask_tensor * 1e8
            _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
            topK = topK.cpu().detach().numpy().tolist()
            predict_items = []
            predict_items.extend(topK)
            tstLocs_np = np.array(
                self.handler.tstLoader.dataset.tstLocs, dtype=object
            )  
            target_items = tstLocs_np[usr.cpu()]  
            precision, recall, NDCG, MRR = computeTopNAccuracy(
                target_items, predict_items, self.topN
            )

            def accumulate(epMetric, metric):
                if epMetric is None:
                    return metric
                else:
                    return [epMetric[j] + metric[j] for j in range(len(metric))]

            
            epRecall = accumulate(epRecall, recall)
            epNdcg = accumulate(epNdcg, NDCG)
            epPrecision = accumulate(epPrecision, precision)
            epMRR = accumulate(epMRR, MRR)
        ret = dict()
        ret["Recall"] = [x / num for x in epRecall]
        ret["NDCG"] = [x / num for x in epNdcg]
        ret["Precision"] = [x / num for x in epPrecision]
        ret["MRR"] = [x / num for x in epMRR]
        return ret
