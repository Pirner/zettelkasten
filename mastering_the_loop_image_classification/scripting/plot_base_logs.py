from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    log_path = '../train_logs.csv'
    df = pd.read_csv(log_path)
    df['epoch'] = range(len(df))

    sns.set_theme()
    # ax = sns.lineplot(data=df, x="epoch", y="train_loss")
    ax = sns.lineplot(data=df, x="epoch", y='train_accuracy')
    ax.legend()
    plt.show()
    exit(0)


if __name__ == '__main__':
    main()
