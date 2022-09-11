# like 1_2_4 but correct recs used in guess loss 

import functools
import glob
import os
import random
import sys

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module

py.arg('--gt_dataset_name', default='dshapes_split_color_vs_ori_size')
py.arg('--gt_dataset_root', default='')
py.arg('--gt_dataset_fn', default='')
py.arg('--gt_dataset_img_folder', default='')
py.arg('--gt_attr_clf_model_path', default='')

py.arg('--output_dir', default='')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--flip', type=bool, default=True)     # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--log_img_every', type=int, default=500) 
py.arg('--write_n_imgs', type=int, default=3)
py.arg('--write_imgs_to_disk', type=bool, default=False)
py.arg('--save_img_to_tb', type=bool, default=True) 
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--guess_loss_weight', type=float, default=0.0)
py.arg('--defence_noise_sigma', type=float, default=0.0)
py.arg('--style_rand_rec_weight', type=float, default=0.0)
py.arg('--style_zero_rec_weight', type=float, default=0.0)
py.arg('--style_idt_rec_weight', type=float, default=0.0)
py.arg('--style_norm_weight', type=float, default=0.0)
py.arg('--gt_data_weight', type=float, default=0.0)
py.arg('--seed', type=int, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples

args = py.args()

dataset_defaults = {
    'dshapes_split_color_vs_ori_size': {
        'gt_dataset_root': '/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/',
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all_all',
        'gt_attr_clf_model_path': '/scratch3/run/better_cg/shapes_keras_pred_model/model_v2'
    },
    'synaction_bg_vs_idt_v2': {
        'gt_dataset_root': '/scratch3/data/guess_disent/synaction_bg_vs_idt_v2/',
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': ''
    },
    'synaction_bg_vs_idt_v3': {
        'gt_dataset_root': '/scratch3/data/guess_disent/synaction_bg_vs_idt_v3/',
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': ''
    }
}

for k, v in dataset_defaults[args.gt_dataset_name].items():
    if getattr(args, k) == '':
        setattr(args, k, v)

tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


@tf.function
def attr_clf_model_predict(imgs):
    return attr_clf_model(imgs)

def predict_attrs(imgs, is_cat_dict):
    preds = attr_clf_model_predict(imgs)
    final_pred_dict = {
        n: (np.argmax(x, axis=1) if is_cat_dict[n] else x[:, 0]) 
        for n, x in zip(attr_clf_model.output_names, preds)
    }
    return final_pred_dict

def update_attr_metrics(metrics, output_dict, to_compare_names, role_names, is_cat_dict):
    for comp_names in to_compare_names:
        attr_preds = {n: predict_attrs(output_dict[n], is_cat_dict) for n in comp_names}
        gen_name = comp_names[0]
        split_name = gen_name[:3]
        other_names = dict(zip(role_names, comp_names[1:]))
        for attr_name in attr_preds[gen_name].keys():
            for cmp_role, cmp_name in other_names.items():
                y_true = attr_preds[cmp_name][attr_name]
                y_pred = attr_preds[gen_name][attr_name]
                metrics[split_name][cmp_role][attr_name].update_state(y_true, y_pred)

def get_disent_fn(model):
    def fn(x, y, **kwargs):
        c = tf.concat([x, y], axis=-1)
        out = model(c, **kwargs)
        return out[..., :-disent_dim], out[..., -disent_dim:]
    return fn


def with_noise(t):
    if args.defence_noise_sigma > 0.0:
        noise = tf.random.normal(t.shape) * args.defence_noise_sigma
        return t + noise
    else:
        return t

# output_dir
folder_suffix = ''
for suf_i in range(1, 20):
    potential_folder = args.output_dir.rstrip("/") + folder_suffix
    if os.path.exists(potential_folder):
        folder_suffix = '_%d' % suf_i
    else:
        output_dir = potential_folder
        py.mkdir(output_dir)
        break
else:
    raise RuntimeError('all potential folders exist! weird!')

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

dataset_file_rows = []
with open(os.path.join(args.gt_dataset_root, args.gt_dataset_fn)) as f:
    for line in f.readlines():
        dataset_file_rows.append([
            os.path.join(args.gt_dataset_root, args.gt_dataset_img_folder, x) for x in line.strip().split(' ')
        ])

random.shuffle(dataset_file_rows)

datasets = [
    data.make_dataset(list(paths_list), args.batch_size, args.load_size, args.crop_size, 
                      training=False, drop_remainder=True, shuffle=False, repeat=1, flip=False) 
    for paths_list in zip(*dataset_file_rows)]

len_dataset = len(dataset_file_rows)
A_B_dataset = tf.data.Dataset.zip(tuple(datasets))
A_B_dataset_test = A_B_dataset.repeat()

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

if args.gt_attr_clf_model_path != '':
    attr_clf_model = tf.keras.models.load_model(args.gt_attr_clf_model_path)

disent_dim = 1
io_dim = 3 + disent_dim
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, io_dim), output_channels=io_dim)
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, io_dim), output_channels=io_dim)

G_A2B_fn = get_disent_fn(G_A2B)
G_B2A_fn = get_disent_fn(G_B2A)

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

if args.guess_loss_weight > 0:
    guess_dim = 6
    D_A_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
    D_B_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
    extra_ckpt_dict = {'D_A_guess': D_A_guess, 'D_B_guess': D_B_guess}
else:
    D_A_guess, D_B_guess = None, None
    extra_ckpt_dict = {}

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


@tf.function
def train_G(A, B, A2B_B_s_gt, B2A_A_s_gt):
    with tf.GradientTape() as t:
        s_zero = tf.zeros((*A.shape[:-1], 1))
        s_a = G_A2B_fn(A, s_zero, training=True)[1]
        s_b = G_B2A_fn(B, s_zero, training=True)[1]
        s_a_rand, s_b_rand = [tf.roll(x, 1, 0) for x in [s_a, s_b]]
        A2B, s_a_nz = G_A2B_fn(A, s_b_rand, training=True)
        B2A, s_b_nz = G_B2A_fn(B, s_a_rand, training=True)
        A2B2A, s_b_rand_rec = G_B2A_fn(with_noise(A2B), with_noise(s_a), training=True)
        B2A2B, s_a_rand_rec = G_A2B_fn(with_noise(B2A), with_noise(s_b), training=True)
        A2A, s_a_idt = G_B2A_fn(A, s_a, training=True)
        B2B, s_b_idt = G_A2B_fn(B, s_b, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        A2B_B_s, _ = G_A2B_fn(A, s_b, training=True)
        B2A_A_s, _ = G_B2A_fn(B, s_a, training=True)

        G_losses = [
            (A2B_g_loss + B2A_g_loss),
            (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight,
            (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight
        ]

        gt_data_loss = (
            cycle_loss_fn(A2B_B_s, A2B_B_s_gt)
            + cycle_loss_fn(B2A_A_s, B2A_A_s_gt))

        style_rec_loss = (
            cycle_loss_fn(s_a_rand, s_a_rand_rec) 
            + cycle_loss_fn(s_b_rand, s_b_rand_rec))

        style_idt_loss =  (
            cycle_loss_fn(s_a, s_a_idt)
            + cycle_loss_fn(s_b, s_b_idt))

        style_zero_rec = (
            cycle_loss_fn(s_a, s_a_nz) 
            + cycle_loss_fn(s_b, s_b_nz))

        style_norm_loss = (
            cycle_loss_fn(s_a, tf.zeros(s_a.shape)) 
            + cycle_loss_fn(s_b, tf.zeros(s_b.shape)))

        A_fwd_guess_logits = D_A_guess(tf.concat([A2B2A, A], axis=-1), training=True)
        A_rev_guess_logits = D_A_guess(tf.concat([A, A2B2A], axis=-1), training=True)
        B_fwd_guess_logits = D_B_guess(tf.concat([B2A2B, B], axis=-1), training=True)
        B_rev_guess_logits = D_B_guess(tf.concat([B, B2A2B], axis=-1), training=True)

        A_fwd_guess_loss, A_rev_guess_loss = d_loss_fn(A_fwd_guess_logits, A_rev_guess_logits)
        B_fwd_guess_loss, B_rev_guess_loss = d_loss_fn(B_fwd_guess_logits, B_rev_guess_logits)

        if args.gt_data_weight:
            G_losses.append(args.gt_data_weight * gt_data_loss)

        if args.style_rand_rec_weight > 0:
            G_losses.append(args.style_rand_rec_weight * style_rec_loss)

        if args.style_idt_rec_weight > 0:
            G_losses.append(args.style_idt_rec_weight * style_idt_loss)

        if args.style_zero_rec_weight > 0:
            G_losses.append(args.style_zero_rec_weight * style_zero_rec)

        if args.style_norm_weight > 0:
            G_losses.append(args.style_norm_weight * style_norm_loss)

        if args.guess_loss_weight > 0:
            G_losses.append(
                (A_fwd_guess_loss + A_rev_guess_loss + B_fwd_guess_loss + B_rev_guess_loss) * args.guess_loss_weight
            )

        G_loss = sum(G_losses)

    G_vars = G_A2B.trainable_variables + G_B2A.trainable_variables
    G_grad = t.gradient(G_loss, G_vars)
    G_optimizer.apply_gradients(zip(G_grad, G_vars))

    summary = {
        'A2B_g_loss': A2B_g_loss, 
        'B2A_g_loss': B2A_g_loss, 
        'A2B2A_cycle_loss': A2B2A_cycle_loss, 
        'B2A2B_cycle_loss': B2A2B_cycle_loss, 
        'A2A_id_loss': A2A_id_loss, 
        'B2B_id_loss': B2B_id_loss,
        'gt_data_loss': gt_data_loss, 
        'A_fwd_guess_loss': A_fwd_guess_loss,
        'A_rev_guess_loss': A_rev_guess_loss,
        'B_fwd_guess_loss': B_fwd_guess_loss,
        'B_rev_guess_loss': B_rev_guess_loss,
        'style_rec_loss': style_rec_loss,
        'style_idt_loss': style_idt_loss,
        'style_zero_rec': style_zero_rec,
        'style_norm_loss': style_norm_loss
    }

    return A2B, B2A, A2B2A, B2A2B, summary


@tf.function
def train_D(A, B, A2B, B2A, A2B2A, B2A2B):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_losses = [
            (A_d_loss + B2A_d_loss),
            (B_d_loss + A2B_d_loss),
            (D_A_gp + D_B_gp) * args.gradient_penalty_weight
        ]

        if args.guess_loss_weight > 0:
            A_fwd_guess_logits = D_A_guess(tf.concat([A2B2A, A], axis=-1), training=True)
            A_rev_guess_logits = D_A_guess(tf.concat([A, A2B2A], axis=-1), training=True)
            B_fwd_guess_logits = D_B_guess(tf.concat([B2A2B, B], axis=-1), training=True)
            B_rev_guess_logits = D_B_guess(tf.concat([B, B2A2B], axis=-1), training=True)

            A_rev_guess_loss, A_fwd_guess_loss = d_loss_fn(A_rev_guess_logits, A_fwd_guess_logits)
            B_rev_guess_loss, B_fwd_guess_loss = d_loss_fn(B_rev_guess_logits, B_fwd_guess_logits)
            D_losses.append(
                (A_fwd_guess_loss + A_rev_guess_loss + B_fwd_guess_loss + B_rev_guess_loss) * args.guess_loss_weight
            )

        D_loss = sum(D_losses)

    D_vars = D_A.trainable_variables + D_B.trainable_variables

    if args.guess_loss_weight > 0:
        D_vars = D_vars + D_A_guess.trainable_variables + D_B_guess.trainable_variables

    D_grad = t.gradient(D_loss, D_vars)
    D_optimizer.apply_gradients(zip(D_grad, D_vars))

    summary = {
        'A_d_loss': A_d_loss + B2A_d_loss, 'B_d_loss': B_d_loss + A2B_d_loss,
        'D_A_gp': D_A_gp, 'D_B_gp': D_B_gp
    }

    if args.guess_loss_weight > 0:
        summary.update({
            'A_fwd_guess_loss': A_fwd_guess_loss,
            'A_rev_guess_loss': A_rev_guess_loss,
            'B_fwd_guess_loss': B_fwd_guess_loss,
            'B_rev_guess_loss': B_rev_guess_loss
        })

    return summary


def train_step(A, B, A2B_B_s_gt, B2A_A_s_gt):
    A2B, B2A, A2B2A, B2A2B, G_loss_dict = train_G(A, B, A2B_B_s_gt, B2A_A_s_gt)

    A2B = A2B_pool(A2B)
    B2A = B2A_pool(B2A)

    D_loss_dict = train_D(A, B, A2B, B2A, A2B2A, B2A2B)

    return G_loss_dict, D_loss_dict


@tf.function
def sample_zs(A, B):
    s_zero = tf.zeros((*A.shape[:-1], 1))
    s_a = G_A2B_fn(A, s_zero, training=True)[1]
    s_b = G_B2A_fn(B, s_zero, training=True)[1]
    s_a_rand, s_b_rand = [tf.roll(x, 1, 0) for x in [s_a, s_b]]
    s_a_perm, s_b_perm = s_a[::-1], s_b[::-1]
    A_perm, B_perm = A[::-1], B[::-1]
    A2B_rand_s = G_A2B_fn(A, s_b_rand, training=False)[0]
    B2A_rand_s = G_B2A_fn(B, s_a_rand, training=False)[0]
    A2B2A = G_B2A_fn(A2B_rand_s, s_a, training=False)[0]
    B2A2B = G_A2B_fn(B2A_rand_s, s_b, training=False)[0]
    A2B2A_perm = G_B2A_fn(A2B_rand_s, s_a_perm, training=False)[0]
    B2A2B_perm = G_A2B_fn(B2A_rand_s, s_b_perm, training=False)[0]
    A2B2A_rand = G_B2A_fn(A2B_rand_s, s_a_rand, training=False)[0]
    B2A2B_rand = G_A2B_fn(B2A_rand_s, s_b_rand, training=False)[0]
    
    A2B_B_s = G_A2B_fn(A, s_b, training=False)[0]
    B2A_A_s = G_B2A_fn(B, s_a, training=False)[0]

    names = ['A2B', 'B2A', 'A2B2A', 'B2A2B', 'A_perm', 'B_perm', 
            'A2B2A_perm', 'B2A2B_perm', 'A2B2A_rand', 'B2A2B_rand',
            'A2B_B_s', 'B2A_A_s']

    values = [A2B_rand_s, B2A_rand_s, A2B2A, B2A2B, A_perm, B_perm, 
              A2B2A_perm, B2A2B_perm, A2B2A_rand, B2A2B_rand,
              A2B_B_s, B2A_A_s]
    
    outputs = dict(zip(names, values))
    return outputs


@tf.function
def sample_grid_zs(A, B):
    bs = A.shape[0]
    s_zero = tf.zeros((*A.shape[:-1], 1))
    s_a = G_A2B_fn(A, s_zero, training=True)[1]
    s_b = G_B2A_fn(B, s_zero, training=True)[1]

    A2B_B_s_grid_flat_list, B2A_A_s_grid_flat_list = [], []
    for k in range(bs):
      A2B_B_s_grid_flat_list.append(
          G_A2B_fn(tf.repeat(A[k, None], bs, 0), s_b, training=False)[0])
      B2A_A_s_grid_flat_list.append(
          G_B2A_fn(tf.repeat(B[k, None], bs, 0), s_a, training=False)[0]
      )
    A2B_B_s_grid_flat = tf.concat(A2B_B_s_grid_flat_list, axis=0)
    B2A_A_s_grid_flat = tf.concat(B2A_A_s_grid_flat_list, axis=0)
    A2B_B_s_grid, B2A_A_s_grid = [
      tf.reshape(x, (bs, bs, *A.shape[1:])) 
      for x in [A2B_B_s_grid_flat, B2A_A_s_grid_flat]]
    return A2B_B_s_grid, B2A_A_s_grid


def get_total_grid(A, B, A2B_B_s_grid):
  bs = A.shape[0]
  img_s = A.shape[1]
  draw_col = A.numpy().reshape(bs*img_s, img_s, 3)
  draw_row = B.numpy().swapaxes(0, 1).reshape(img_s, bs*img_s, 3)
  draw_grid = A2B_B_s_grid.numpy().swapaxes(1, 2).reshape(img_s*bs, img_s*bs, 3)
  total_grid_items = [[np.zeros((img_s, img_s, 3)), draw_row], [draw_col, draw_grid]]
  total_grid = np.concatenate([np.concatenate(x, axis=1) for x in total_grid_items], axis=0)
  return total_grid


def get_AB_zs_grids(A, B):
    A2B_B_s_grid, B2A_A_s_grid = sample_grid_zs(A, B)
    grid_A = get_total_grid(A, B, A2B_B_s_grid)
    grid_B = get_total_grid(B, A, B2A_A_s_grid)
    return grid_A, grid_B


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt,
                                **extra_ckpt_dict),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5,
                           keep_checkpoint_every_n_hours=2)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries'))

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# otherwise might have issues with swapping / random
assert args.batch_size > 3

is_categorical_attr = {
    'f_hue': True, 'w_hue': True, 'o_hue': True, 
    'size': False, 'shape': True, 'ori': False}

to_compare_names = [('A2B_B_s', 'A', 'B', 'A2B_B_s_gt'), ('B2A_A_s', 'B', 'A', 'B2A_A_s_gt')]
role_names = ['source', 'guide', 'gt']

def get_metric_dict():
    result = {}
    for attr_n, is_cat in is_categorical_attr.items():
        result[attr_n] = (
            tf.keras.metrics.Accuracy() if is_cat 
            else tf.keras.metrics.MeanAbsoluteError())

    return result

# main loop
with train_summary_writer.as_default():
    with open(py.join(output_dir, 'settings.yml')) as f:
        lines = [' '.join(sys.argv) + '\n\n'] + f.readlines()
        tf.summary.text('args', ''.join(['    ' + x for x in lines]), step=0)

    # source code
    with open(sys.argv[0]) as f:
        tf.summary.text('code', ''.join(['    ' + x for x in f.readlines()]), step=0)

    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B, A2B_B_s_gt, B2A_A_s_gt in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B, A2B_B_s_gt, B2A_A_s_gt)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % args.log_img_every == 0:
                metrics = {
                    split: {rn: get_metric_dict() for rn in role_names} for split in ['A2B', 'B2A']
                }

                imgs = []
                grid_imgs = []
                test_iter = iter(A_B_dataset_test)
                for img_idx in range(args.write_n_imgs):
                    A, B, A2B_B_s_gt, B2A_A_s_gt = next(test_iter)
                    output_sample_dict = sample_zs(A, B)
                    output_sample_dict.update(
                        dict(zip(['A', 'B', 'A2B_B_s_gt', 'B2A_A_s_gt'], [A, B, A2B_B_s_gt, B2A_A_s_gt])))
                    img_names_to_merge = [
                        'A', 'A2B', 'A2B2A', 'A2B2A_rand', 'A_perm', 'A2B2A_perm', 'A2B_B_s', 'A2B_B_s_gt',
                        'B', 'B2A', 'B2A2B', 'B2A2B_rand', 'B_perm', 'B2A2B_perm', 'B2A_A_s', 'B2A_A_s_gt']
                    
                    for k in range(3):
                        single_img_to_merge = [output_sample_dict[n][k, None] for n in img_names_to_merge]
                        img = im.immerge(np.concatenate(single_img_to_merge, axis=0), n_rows=2)
                        imgs.append((img + 1) * 0.5)

                    if args.gt_attr_clf_model_path != '':
                        update_attr_metrics(metrics, output_sample_dict, to_compare_names, role_names, is_categorical_attr)

                    grid_imgs.append(get_AB_zs_grids(A, B))

                    if args.write_imgs_to_disk:
                        fn = 'iter-%09d-%d.jpg' % (G_optimizer.iterations.numpy(), img_idx)
                        im.imwrite(img, py.join(sample_dir, fn))

                if args.save_img_to_tb:
                    tf.summary.image(
                        'img', imgs, 
                        step=G_optimizer.iterations, max_outputs=args.write_n_imgs * 3)

                    if grid_imgs:
                        for name, grid in zip(['A', 'B'], zip(*grid_imgs)):
                            tf.summary.image('grid-%s' % name, 
                                [x * 0.5 + 0.5 for x in grid], 
                                step=G_optimizer.iterations,
                                max_outputs=args.write_n_imgs)

                if args.gt_attr_clf_model_path != '':
                    for split_name, split_metrics in metrics.items():
                        for cmp_name, attr_metrics in split_metrics.items():
                            for attr_name, metric in attr_metrics.items():
                                name = 'attr/%s-%s-%s' % (split_name, cmp_name, attr_name)
                                if isinstance(metric, tf.keras.metrics.Accuracy):
                                    value = 1.0 - metric.result()
                                else:
                                    value = metric.result()
                                tf.summary.scalar(name, value, step=G_optimizer.iterations)

        # save checkpoint
        checkpoint.save(ep)
