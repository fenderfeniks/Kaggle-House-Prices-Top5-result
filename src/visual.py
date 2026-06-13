import matplotlib.pyplot as plt



def visualise_mistakes(model, y_true, y_pred, residuals):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_true, y_pred, alpha=0.3)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--')
    axes[0].set_title(f'{model.__class__.__name__} — Predicted vs Actual')

    axes[1].scatter(y_pred, residuals, alpha=0.3)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title('Residuals vs Predicted')

    axes[2].hist(residuals, bins=50)
    axes[2].set_title('Распределение остатков')

    plt.tight_layout()
    plt.show()

def visualise_for_target(df, y_train):
    fig, ax = plt.subplots()
    ax.scatter(df, y_train)
    ax.set_title(df.name)
    plt.show()