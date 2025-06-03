class MCEvaluator:
    def __init__(self):
        pass

    def score_single(self, pred_ans, real_ans):
        pos = max(0, pred_ans.find("boxed{"))
        pred = "a"
        for i in range(pos, len(pred_ans)):
            if pred_ans[i] >= "A" and pred_ans[i] <= "Z":
                pred = pred_ans[i]
                break

        if "###" in real_ans:
            real = real_ans.split("###")[0].strip()
        else:
            pos = real_ans.find("boxed{")
            real = "b"
            for i in range(pos, len(real_ans)):
                if real_ans[i] >= "A" and real_ans[i] <= "D":
                    real = real_ans[i]
                    break

        return pred == real

    def score(self, pred_ans, real_ans):
        score = [self.score_single(p, r) for p, r in zip(pred_ans, real_ans)]
        return score
