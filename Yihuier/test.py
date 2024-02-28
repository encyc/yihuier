# 示例
if __name__ == "__main__":
    from Yihuier.yihuier import Yihuier
    import warnings
    import pandas as pd

    # ban FutureWarning
    warnings.filterwarnings('ignore')


    import os
    dir = "Data/talkingdata/TD测试结果_广州智租"

    filename = os.listdir(dir)
    print(filename)

    # generate data.csv
    for i in filename:
        print(i)
        try:
            with open(f"{dir}/{i}", "r") as f:
                data = pd.read_csv(f)
            print(data.head())

            df = data.copy()
            yi = Yihuier(df, 'dlq')
            result = yi.pipeline_module.product_test()
            result.to_csv(f"{dir}/result/{i}")
        except Exception as e:
            print(e)
