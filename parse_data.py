import re
import pickle
from functools import partial

import torch
import numpy as np
from scipy.stats import ttest_rel


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def parse_basic_table_data(exp_data, map_data):
    models = dict()
    exp_logging_values = exp_data["logging"]
    exp_logging_values = [value.item() for value in exp_logging_values]
    map_logging_values = map_data["logging"]
    map_logging_values = [value.item() for value in map_logging_values]

    for model in exp_data.keys():
        parsed_vals = dict(mean=dict(), std=dict())
        parsed_vals["pval two-sided"] = dict()

        for stat, values in zip(["exp", "map"], [exp_data[model], map_data[model]]):
            parsed_vals["mean"][stat] = np.mean(values)
            parsed_vals["std"][stat] = np.std(values)

            if stat == "exp":
                parsed_vals["pval two-sided"][stat] = ttest_rel(exp_logging_values, values, alternative="two-sided").pvalue
            elif stat == "map":
                parsed_vals["pval two-sided"][stat] = ttest_rel(map_logging_values, values, alternative="two-sided").pvalue
        models[model] = parsed_vals

    return models


if __name__=="__main__":
    from torch import tensor
    from results_to_latex import to_latex_table
    data = {'logging': [tensor(1.5279), tensor(1.5369), tensor(1.5476), tensor(1.5097), tensor(1.5809), tensor(1.5250), tensor(1.5133), tensor(1.5407), tensor(1.5124), tensor(1.5026)], 'logger-og': [1.5366257285419435452, 1.5336465491805278097, 1.5462175334295371683, 1.5142160127750614926, 1.575286494530345022, 1.5205836963099660141, 1.5116993846081508044, 1.5385773258528476445, 1.5160383344762992484, 1.5022293047223008364], 'nn-noreg 4-4': [tensor(2.1865), tensor(1.9793), tensor(2.3990), tensor(2.1214), tensor(1.9418), tensor(2.0589), tensor(2.0384), tensor(2.4375), tensor(2.3306), tensor(2.4316)], 'nn-soft 4-4': [tensor(4.2931), tensor(4.8770), tensor(4.2090), tensor(4.8353), tensor(3.7183), tensor(3.9330), tensor(4.3781), tensor(4.4827), tensor(4.5114), tensor(4.6125)], 'nn-hard 4-4': [tensor(2.6744), tensor(2.3235), tensor(2.8024), tensor(2.6951), tensor(2.4911), tensor(2.7360), tensor(2.1634), tensor(2.7521), tensor(2.5204), tensor(2.5021)], 'nn-noreg 8-8': [tensor(1.9790), tensor(1.9023), tensor(2.0834), tensor(2.1950), tensor(1.8919), tensor(1.8765), tensor(2.1065), tensor(2.3395), tensor(2.5396), tensor(2.0105)], 'nn-soft 8-8': [tensor(4.6136), tensor(4.1196), tensor(3.7314), tensor(4.8315), tensor(4.8134), tensor(4.5901), tensor(4.3167), tensor(4.8327), tensor(4.4895), tensor(4.4527)], 'nn-hard 8-8': [tensor(2.6101), tensor(2.4557), tensor(2.4587), tensor(2.4699), tensor(2.4785), tensor(2.4727), tensor(2.3268), tensor(2.5395), tensor(2.7490), tensor(2.6442)], 'nn-noreg 16-16': [tensor(2.1118), tensor(2.0710), tensor(1.9867), tensor(1.9824), tensor(1.9392), tensor(1.9459), tensor(1.8311), tensor(1.9239), tensor(2.0488), tensor(1.7545)], 'nn-soft 16-16': [tensor(4.4612), tensor(4.8648), tensor(4.1628), tensor(4.7720), tensor(4.8974), tensor(4.7865), tensor(4.8554), tensor(3.3833), tensor(3.9353), tensor(4.8388)], 'nn-hard 16-16': [tensor(2.4475), tensor(2.6438), tensor(2.6291), tensor(2.5687), tensor(2.6628), tensor(2.5242), tensor(2.5282), tensor(2.6090), tensor(2.3885), tensor(2.4973)], 'nn-noreg 32-16': [tensor(2.0289), tensor(1.8610), tensor(2.0895), tensor(2.0231), tensor(1.9876), tensor(1.9854), tensor(1.9025), tensor(2.2961), tensor(2.2381), tensor(1.9004)], 'nn-soft 32-16': [tensor(3.5165), tensor(4.8323), tensor(3.6161), tensor(4.6202), tensor(4.2429), tensor(4.8722), tensor(4.3037), tensor(4.8202), tensor(4.8603), tensor(4.3288)], 'nn-hard 32-16': [tensor(2.6054), tensor(2.5117), tensor(2.4931), tensor(2.4305), tensor(2.6466), tensor(2.6465), tensor(2.5099), tensor(2.6479), tensor(2.6225), tensor(2.5742)], 'nn-noreg 32-32': [tensor(1.8797), tensor(1.8299), tensor(1.9831), tensor(1.8389), tensor(1.8841), tensor(1.8625), tensor(1.7362), tensor(1.9902), tensor(1.9589), tensor(1.7825)], 'nn-soft 32-32': [tensor(4.8775), tensor(4.5098), tensor(4.7306), tensor(4.3418), tensor(4.0912), tensor(4.3341), tensor(4.2811), tensor(4.8744), tensor(4.2714), tensor(4.8596)], 'nn-hard 32-32': [tensor(2.6223), tensor(2.6133), tensor(2.4645), tensor(2.4486), tensor(2.4676), tensor(2.6292), tensor(2.5832), tensor(2.6936), tensor(2.4928), tensor(2.5099)], 'nn-noreg 64-32': [tensor(1.8869), tensor(1.7367), tensor(1.9313), tensor(1.9477), tensor(1.9767), tensor(1.7809), tensor(1.8398), tensor(1.9357), tensor(1.8942), tensor(1.9703)], 'nn-soft 64-32': [tensor(3.9240), tensor(3.9390), tensor(4.8683), tensor(4.7548), tensor(4.8916), tensor(4.4448), tensor(3.7923), tensor(4.4242), tensor(4.3803), tensor(4.8349)], 'nn-hard 64-32': [tensor(2.5177), tensor(2.5919), tensor(2.6187), tensor(2.6122), tensor(2.6603), tensor(2.5125), tensor(2.5150), tensor(2.7034), tensor(2.5779), tensor(2.3234)], 'nn-noreg 64-64': [tensor(1.8018), tensor(1.7792), tensor(1.8344), tensor(1.7933), tensor(1.8172), tensor(1.7491), tensor(1.7489), tensor(1.8683), tensor(1.8552), tensor(1.7486)], 'nn-soft 64-64': [tensor(4.8461), tensor(4.8754), tensor(4.8188), tensor(4.6130), tensor(4.0717), tensor(4.0135), tensor(4.2059), tensor(4.3457), tensor(4.3927), tensor(4.3287)], 'nn-hard 64-64': [tensor(2.6363), tensor(2.5687), tensor(2.5262), tensor(2.5708), tensor(2.6147), tensor(2.4614), tensor(2.4096), tensor(2.5431), tensor(2.5477), tensor(2.5775)], 'nn-noreg 128-128': [tensor(1.7761), tensor(1.6812), tensor(1.8210), tensor(1.7606), tensor(1.8416), tensor(1.7889), tensor(1.7572), tensor(1.8279), tensor(1.7325), tensor(1.7732)], 'nn-soft 128-128': [tensor(4.3712), tensor(4.8224), tensor(4.4514), tensor(4.8578), tensor(4.8022), tensor(4.8929), tensor(4.4970), tensor(4.1410), tensor(4.7112), tensor(4.9083)], 'nn-hard 128-128': [tensor(2.3174), tensor(2.5052), tensor(2.4260), tensor(2.3954), tensor(2.6843), tensor(2.4779), tensor(2.5664), tensor(2.4886), tensor(2.4664), tensor(2.5329)], 'crf': [tensor(1.0629), tensor(1.0883), tensor(1.0780), tensor(1.0720), tensor(1.0766), tensor(1.0806), tensor(1.0720), tensor(1.0815), tensor(1.0734), tensor(1.0791)], 'prm': [tensor(1.1032), tensor(1.2275), tensor(1.0878), tensor(1.1031), tensor(1.1483), tensor(1.2547), tensor(1.1863), tensor(1.1119), tensor(1.1004), tensor(1.2097)], 'prm-og': [1.1033125909467577062, 1.2273758044503332204, 1.0877971522100900873, 1.1033958672291573107, 1.1483759729611864261, 1.254776452533203787, 1.1861663244869947182, 1.1118767426123576095, 1.1005475613974101946, 1.2095962806254088636], 'erm': [tensor(1.1160), tensor(1.2409), tensor(1.0923), tensor(1.1832), tensor(1.2204), tensor(1.3925), tensor(1.1706), tensor(1.1100), tensor(1.0976), tensor(1.2430)], 'erm-og': [1.1163497832992946416, 1.2408985954695886502, 1.0922818665456298161, 1.1829394076582623342, 1.2204927927079882945, 1.3924817348954531744, 1.1705662870762142787, 1.1097923937234593341, 1.0971840893728958476, 1.2434770877016311145], 'maj': [tensor(1.0861), tensor(1.0814), tensor(1.0862), tensor(1.0862), tensor(1.0862), tensor(1.0861), tensor(1.0862), tensor(1.0862), tensor(1.0862), tensor(1.0861)], 'maj-og': [1.0861226955433001787, 1.0825784581062800168, 1.086165865022254695, 1.0861240574885631469, 1.0861538296966091615, 1.0861175715829262263, 1.086169869204428784, 1.0861930533599987831, 1.0861343849184951049, 1.086121786044087763], 'majerm': [tensor(2.9267), tensor(1.0865), tensor(1.0862), tensor(1.0862), tensor(1.0881), tensor(1.7400), tensor(1.0862), tensor(1.6898), tensor(1.1442), tensor(1.7516)], 'majerm-og': [2.926719758537501372, 1.0866329383829524804, 1.0861672248511658951, 1.0861833330840987118, 1.0880815035194920271, 1.7399919779035259219, 1.0861468712481750934, 1.6897898572821212061, 1.1427296873904421763, 1.7517853843531619924]}
    data_2 = {'logging': [tensor(1.5279), tensor(1.5369), tensor(1.5476), tensor(1.5097), tensor(1.5809), tensor(1.5250), tensor(1.5133), tensor(1.5407), tensor(1.5124), tensor(1.5026)], 'logger-og': [1.5366257285419435452, 1.5336465491805278097, 1.5462175334295371683, 1.5142160127750614926, 1.575286494530345022, 1.5205836963099660141, 1.5116993846081508044, 1.5385773258528476445, 1.5160383344762992484, 1.5022293047223008364], 'nn-noreg 4-4': [tensor(2.1865), tensor(1.9793), tensor(2.3990), tensor(2.1214), tensor(1.9418), tensor(2.0589), tensor(2.0384), tensor(2.4375), tensor(2.3306), tensor(2.4316)], 'nn-soft 4-4': [tensor(4.2931), tensor(4.8770), tensor(4.2090), tensor(4.8353), tensor(3.7183), tensor(3.9330), tensor(4.3781), tensor(4.4827), tensor(4.5114), tensor(4.6125)], 'nn-hard 4-4': [tensor(2.6744), tensor(2.3235), tensor(2.8024), tensor(2.6951), tensor(2.4911), tensor(2.7360), tensor(2.1634), tensor(2.7521), tensor(2.5204), tensor(2.5021)], 'nn-noreg 8-8': [tensor(1.9790), tensor(1.9023), tensor(2.0834), tensor(2.1950), tensor(1.8919), tensor(1.8765), tensor(2.1065), tensor(2.3395), tensor(2.5396), tensor(2.0105)], 'nn-soft 8-8': [tensor(4.6136), tensor(4.1196), tensor(3.7314), tensor(4.8315), tensor(4.8134), tensor(4.5901), tensor(4.3167), tensor(4.8327), tensor(4.4895), tensor(4.4527)], 'nn-hard 8-8': [tensor(2.6101), tensor(2.4557), tensor(2.4587), tensor(2.4699), tensor(2.4785), tensor(2.4727), tensor(2.3268), tensor(2.5395), tensor(2.7490), tensor(2.6442)], 'nn-noreg 16-16': [tensor(2.1118), tensor(2.0710), tensor(1.9867), tensor(1.9824), tensor(1.9392), tensor(1.9459), tensor(1.8311), tensor(1.9239), tensor(2.0488), tensor(1.7545)], 'nn-soft 16-16': [tensor(4.4612), tensor(4.8648), tensor(4.1628), tensor(4.7720), tensor(4.8974), tensor(4.7865), tensor(4.8554), tensor(3.3833), tensor(3.9353), tensor(4.8388)], 'nn-hard 16-16': [tensor(2.4475), tensor(2.6438), tensor(2.6291), tensor(2.5687), tensor(2.6628), tensor(2.5242), tensor(2.5282), tensor(2.6090), tensor(2.3885), tensor(2.4973)], 'nn-noreg 32-16': [tensor(2.0289), tensor(1.8610), tensor(2.0895), tensor(2.0231), tensor(1.9876), tensor(1.9854), tensor(1.9025), tensor(2.2961), tensor(2.2381), tensor(1.9004)], 'nn-soft 32-16': [tensor(3.5165), tensor(4.8323), tensor(3.6161), tensor(4.6202), tensor(4.2429), tensor(4.8722), tensor(4.3037), tensor(4.8202), tensor(4.8603), tensor(4.3288)], 'nn-hard 32-16': [tensor(2.6054), tensor(2.5117), tensor(2.4931), tensor(2.4305), tensor(2.6466), tensor(2.6465), tensor(2.5099), tensor(2.6479), tensor(2.6225), tensor(2.5742)], 'nn-noreg 32-32': [tensor(1.8797), tensor(1.8299), tensor(1.9831), tensor(1.8389), tensor(1.8841), tensor(1.8625), tensor(1.7362), tensor(1.9902), tensor(1.9589), tensor(1.7825)], 'nn-soft 32-32': [tensor(4.8775), tensor(4.5098), tensor(4.7306), tensor(4.3418), tensor(4.0912), tensor(4.3341), tensor(4.2811), tensor(4.8744), tensor(4.2714), tensor(4.8596)], 'nn-hard 32-32': [tensor(2.6223), tensor(2.6133), tensor(2.4645), tensor(2.4486), tensor(2.4676), tensor(2.6292), tensor(2.5832), tensor(2.6936), tensor(2.4928), tensor(2.5099)], 'nn-noreg 64-32': [tensor(1.8869), tensor(1.7367), tensor(1.9313), tensor(1.9477), tensor(1.9767), tensor(1.7809), tensor(1.8398), tensor(1.9357), tensor(1.8942), tensor(1.9703)], 'nn-soft 64-32': [tensor(3.9240), tensor(3.9390), tensor(4.8683), tensor(4.7548), tensor(4.8916), tensor(4.4448), tensor(3.7923), tensor(4.4242), tensor(4.3803), tensor(4.8349)], 'nn-hard 64-32': [tensor(2.5177), tensor(2.5919), tensor(2.6187), tensor(2.6122), tensor(2.6603), tensor(2.5125), tensor(2.5150), tensor(2.7034), tensor(2.5779), tensor(2.3234)], 'nn-noreg 64-64': [tensor(1.8018), tensor(1.7792), tensor(1.8344), tensor(1.7933), tensor(1.8172), tensor(1.7491), tensor(1.7489), tensor(1.8683), tensor(1.8552), tensor(1.7486)], 'nn-soft 64-64': [tensor(4.8461), tensor(4.8754), tensor(4.8188), tensor(4.6130), tensor(4.0717), tensor(4.0135), tensor(4.2059), tensor(4.3457), tensor(4.3927), tensor(4.3287)], 'nn-hard 64-64': [tensor(2.6363), tensor(2.5687), tensor(2.5262), tensor(2.5708), tensor(2.6147), tensor(2.4614), tensor(2.4096), tensor(2.5431), tensor(2.5477), tensor(2.5775)], 'nn-noreg 128-128': [tensor(1.7761), tensor(1.6812), tensor(1.8210), tensor(1.7606), tensor(1.8416), tensor(1.7889), tensor(1.7572), tensor(1.8279), tensor(1.7325), tensor(1.7732)], 'nn-soft 128-128': [tensor(4.3712), tensor(4.8224), tensor(4.4514), tensor(4.8578), tensor(4.8022), tensor(4.8929), tensor(4.4970), tensor(4.1410), tensor(4.7112), tensor(4.9083)], 'nn-hard 128-128': [tensor(2.3174), tensor(2.5052), tensor(2.4260), tensor(2.3954), tensor(2.6843), tensor(2.4779), tensor(2.5664), tensor(2.4886), tensor(2.4664), tensor(2.5329)], 'crf': [tensor(1.0629), tensor(1.0883), tensor(1.0780), tensor(1.0720), tensor(1.0766), tensor(1.0806), tensor(1.0720), tensor(1.0815), tensor(1.0734), tensor(1.0791)], 'prm': [tensor(1.1032), tensor(1.2275), tensor(1.0878), tensor(1.1031), tensor(1.1483), tensor(1.2547), tensor(1.1863), tensor(1.1119), tensor(1.1004), tensor(1.2097)], 'prm-og': [1.1033125909467577062, 1.2273758044503332204, 1.0877971522100900873, 1.1033958672291573107, 1.1483759729611864261, 1.254776452533203787, 1.1861663244869947182, 1.1118767426123576095, 1.1005475613974101946, 1.2095962806254088636], 'erm': [tensor(1.1160), tensor(1.2409), tensor(1.0923), tensor(1.1832), tensor(1.2204), tensor(1.3925), tensor(1.1706), tensor(1.1100), tensor(1.0976), tensor(1.2430)], 'erm-og': [1.1163497832992946416, 1.2408985954695886502, 1.0922818665456298161, 1.1829394076582623342, 1.2204927927079882945, 1.3924817348954531744, 1.1705662870762142787, 1.1097923937234593341, 1.0971840893728958476, 1.2434770877016311145], 'maj': [tensor(1.0861), tensor(1.0814), tensor(1.0862), tensor(1.0862), tensor(1.0862), tensor(1.0861), tensor(1.0862), tensor(1.0862), tensor(1.0862), tensor(1.0861)], 'maj-og': [1.0861226955433001787, 1.0825784581062800168, 1.086165865022254695, 1.0861240574885631469, 1.0861538296966091615, 1.0861175715829262263, 1.086169869204428784, 1.0861930533599987831, 1.0861343849184951049, 1.086121786044087763], 'majerm': [tensor(2.9267), tensor(1.0865), tensor(1.0862), tensor(1.0862), tensor(1.0881), tensor(1.7400), tensor(1.0862), tensor(1.6898), tensor(1.1442), tensor(1.7516)], 'majerm-og': [2.926719758537501372, 1.0866329383829524804, 1.0861672248511658951, 1.0861833330840987118, 1.0880815035194920271, 1.7399919779035259219, 1.0861468712481750934, 1.6897898572821212061, 1.1427296873904421763, 1.7517853843531619924]}
    data = parse_basic_table_data(data, data_2)
    table = to_latex_table(data, ["map", "exp"])
    print(table)