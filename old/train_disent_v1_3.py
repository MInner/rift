# splitting the mapping into two parts to avoid dependence

import sys
import functools
import glob
import os
import random

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


py.arg('--gt_dataset_root', default='/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/')
py.arg('--gt_dataset_fn', default='gt.txt')
py.arg('--gt_dataset_img_folder', default='all_all')
py.arg('--gt_attr_clf_model_path', default='/scratch3/run/better_cg/shapes_keras_pred_model/model_v2')

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
py.arg('--style_rec_weight', type=float, default=0.0)
py.arg('--style_idt_weight', type=float, default=0.0)
py.arg('--style_norm_weight', type=float, default=0.0)
py.arg('--gt_data_weight', type=float, default=0.0)
py.arg('--shuffle_seed', type=int, default=10)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()


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

def get_disent_fn(model_ab, model_as):
    def fn(x, y, **kwargs):
        out = model_ab(tf.concat([x, y], axis=-1), **kwargs)
        a_style = model_as(x)
        return out, a_style
    return fn


def with_noise(t):
    if args.defence_noise_sigma > 0.0:
        noise = tf.random.normal(t.shape) * args.defence_noise_sigma
        return t + noise
    else:
        return t

# output_dir
output_dir = args.output_dir
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

dataset_file_rows = []
with open(os.path.join(args.gt_dataset_root, args.gt_dataset_fn)) as f:
    for line in f.readlines():
        dataset_file_rows.append([
            os.path.join(args.gt_dataset_root, args.gt_dataset_img_folder, x) for x in line.strip().split(' ')
        ])

random.Random(args.shuffle_seed).shuffle(dataset_file_rows)

datasets = [
    data.make_dataset(list(paths_list), args.batch_size, args.load_size, args.crop_size, 
                      training=False, drop_remainder=True, shuffle=False, repeat=1, flip=False) 
    for paths_list in zip(*dataset_file_rows)]

len_dataset = len(dataset_file_rows)
A_B_dataset = tf.data.Dataset.zip(tuple(datasets))
A_B_dataset_test = A_B_dataset.repeat()

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

attr_clf_model = tf.keras.models.load_model(args.gt_attr_clf_model_path)

img_dims, style_dims = 3, 1
input_dims = img_dims + style_dims
G_A2B_m = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, input_dims), output_channels=img_dims)
G_B2A_m = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, input_dims), output_channels=img_dims)
Es_A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, img_dims), output_channels=style_dims)
Es_B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, img_dims), output_channels=style_dims)

G_A2B_fn = get_disent_fn(G_A2B_m, Es_A)
G_B2A_fn = get_disent_fn(G_B2A_m, Es_B)

get_generator_vars = lambda: sum([x.trainable_variables for x in [G_A2B_m, G_B2A_m, Es_A, Es_B]], [])

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

if args.guess_loss_weight > 0:
    guess_dim = 6
    D_A_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
    D_B_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
else:
    D_A_guess, D_B_guess = None, None

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
        s_a_rand = tf.random.normal((*A.shape[:-1], 1))
        s_b_rand = tf.random.normal((*A.shape[:-1], 1))
        A2B, s_a = G_A2B_fn(A, s_b_rand, training=True)
        B2A, s_b = G_B2A_fn(B, s_a_rand, training=True)
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

        G_losses.append(args.gt_data_weight * gt_data_loss)

        if args.style_rec_weight > 0:
            style_rec_loss = (
                cycle_loss_fn(s_a_rand, s_a_rand_rec) 
                + cycle_loss_fn(s_b_rand, s_b_rand_rec))
            
            G_losses.append(args.style_rec_weight * style_rec_loss)

        if args.style_idt_weight > 0:
            style_idt_loss =  (
                cycle_loss_fn(s_a, s_a_idt)
                + cycle_loss_fn(s_b, s_b_idt))

            G_losses.append(args.style_idt_weight * style_idt_loss)

        if args.style_norm_weight > 0:
            style_norm_loss = (
                cycle_loss_fn(s_a, tf.zeros(s_a.shape)) 
                + cycle_loss_fn(s_b, tf.zeros(s_b.shape)))

            G_losses.append(args.style_norm_weight * style_norm_loss)

        if args.guess_loss_weight > 0:
            A_fwd_guess_logits = D_A_guess(tf.concat([A2B2A, A], axis=-1), training=True)
            A_rev_guess_logits = D_A_guess(tf.concat([A, A2B2A], axis=-1), training=True)
            B_fwd_guess_logits = D_B_guess(tf.concat([B2A2B, B], axis=-1), training=True)
            B_rev_guess_logits = D_B_guess(tf.concat([B, B2A2B], axis=-1), training=True)

            A_fwd_guess_loss, A_rev_guess_loss = d_loss_fn(A_fwd_guess_logits, A_rev_guess_logits)
            B_fwd_guess_loss, B_rev_guess_loss = d_loss_fn(B_fwd_guess_logits, B_rev_guess_logits)
            G_losses.append(
                (A_fwd_guess_loss + A_rev_guess_loss + B_fwd_guess_loss + B_rev_guess_loss) * args.guess_loss_weight
            )

        G_loss = sum(G_losses)

    G_vars = get_generator_vars()
    G_grad = t.gradient(G_loss, G_vars)
    G_optimizer.apply_gradients(zip(G_grad, G_vars))
    summary = {
        'A2B_g_loss': A2B_g_loss, 'B2A_g_loss': B2A_g_loss, 
        'A2B2A_cycle_loss': A2B2A_cycle_loss, 'B2A2B_cycle_loss': B2A2B_cycle_loss, 
        'A2A_id_loss': A2A_id_loss, 'B2B_id_loss': B2B_id_loss}

    summary['gt_data_loss'] = gt_data_loss
    
    if args.guess_loss_weight > 0:
        summary.update({
            'A_fwd_guess_loss': A_fwd_guess_loss,
            'A_rev_guess_loss': A_rev_guess_loss,
            'B_fwd_guess_loss': B_fwd_guess_loss,
            'B_rev_guess_loss': B_rev_guess_loss,
        })

    if args.style_rec_weight > 0:
        summary['style_rec_loss'] = style_rec_loss

    if args.style_idt_weight > 0:
        summary['style_idt_loss'] = style_idt_loss

    if args.style_norm_weight > 0:
        summary['style_norm_loss'] = style_norm_loss

    return A2B, B2A, summary


@tf.function
def train_D(A, B, A2B, B2A):
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
            s_a_rand = tf.random.normal((*A.shape[:-1], 1))
            s_b_rand = tf.random.normal((*A.shape[:-1], 1))
            A2B2A, _ = G_B2A_fn(with_noise(A2B), s_a_rand, training=True)
            B2A2B, _ = G_A2B_fn(with_noise(B2A), s_b_rand, training=True)
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
    A2B, B2A, G_loss_dict = train_G(A, B, A2B_B_s_gt, B2A_A_s_gt)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A_perm, B_perm = A[::-1], B[::-1]
    s_a_rand = tf.random.normal((*A.shape[:-1], 1))
    s_b_rand = tf.random.normal((*A.shape[:-1], 1))
    A2B_rand_s, s_a = G_A2B_fn(A, s_b_rand, training=False)
    B2A_rand_s, s_b = G_B2A_fn(B, s_a_rand, training=False)
    A2B2A, _ = G_B2A_fn(A2B_rand_s, s_a, training=False)
    B2A2B, _ = G_A2B_fn(B2A_rand_s, s_b, training=False)
    A2B2A_perm, _ = G_B2A_fn(A2B_rand_s, s_a[::-1], training=False)
    B2A2B_perm, _ = G_A2B_fn(B2A_rand_s, s_b[::-1], training=False)
    A2B2A_rand, _ = G_B2A_fn(A2B_rand_s, s_a_rand, training=False)
    B2A2B_rand, _ = G_A2B_fn(B2A_rand_s, s_b_rand, training=False)
    
    A2B_B_s, _ = G_A2B_fn(A, s_b, training=False)
    B2A_A_s, _ = G_B2A_fn(B, s_a, training=False)

    names = ['A2B', 'B2A', 'A2B2A', 'B2A2B', 'A_perm', 'B_perm', 
            'A2B2A_perm', 'B2A2B_perm', 'A2B2A_rand', 'B2A2B_rand',
            'A2B_B_s', 'B2A_A_s']

    values = [A2B_rand_s, B2A_rand_s, A2B2A, B2A2B, A_perm, B_perm, 
              A2B2A_perm, B2A2B_perm, A2B2A_rand, B2A2B_rand,
              A2B_B_s, B2A_A_s]
    
    outputs = dict(zip(names, values))
    return outputs


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B_m=G_A2B_m,
                                G_B2A_m=G_B2A_m,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

assert args.batch_size > 1

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
                for img_idx in range(args.write_n_imgs):
                    A, B, A2B_B_s_gt, B2A_A_s_gt = next(test_iter)
                    output_sample_dict = sample(A, B)
                    output_sample_dict.update(
                        dict(zip(['A', 'B', 'A2B_B_s_gt', 'B2A_A_s_gt'], [A, B, A2B_B_s_gt, B2A_A_s_gt])))
                    img_names_to_merge = [
                        'A', 'A2B', 'A2B2A', 'A2B2A_rand', 'A_perm', 'A2B2A_perm', 'A2B_B_s', 'A2B_B_s_gt',
                        'B', 'B2A', 'B2A2B', 'B2A2B_rand', 'B_perm', 'B2A2B_perm', 'B2A_A_s', 'B2A_A_s_gt']

                    single_img_to_merge = [output_sample_dict[n][0, None] for n in img_names_to_merge]
                    img = im.immerge(np.concatenate(single_img_to_merge, axis=0), n_rows=2)
                    imgs.append((img + 1) * 0.5)

                    update_attr_metrics(metrics, output_sample_dict, to_compare_names, role_names, is_categorical_attr)

                    if args.write_imgs_to_disk:
                        fn = 'iter-%09d-%d.jpg' % (G_optimizer.iterations.numpy(), img_idx)
                        im.imwrite(img, py.join(sample_dir, fn))
                    
                if args.save_img_to_tb:
                    tf.summary.image('img', imgs, step=G_optimizer.iterations, max_outputs=args.write_n_imgs)

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
