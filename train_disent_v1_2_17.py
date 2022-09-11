# like 1_2_16 with hidden attr pred

import functools
import glob
import os
import random
import sys
from collections import defaultdict

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
py.arg('--gt_attr_clf_input_size', default=64)
py.arg('--shuffle_supervised_pairs_buffer', type=int, default=0)

py.arg('--gt_main_dataset_root', default='/scratch3/data/guess_disent/')
py.arg('--gt_main_attr_clf_model_root', default='/scratch3/run/honest_disent/attr_models/')

py.arg('--output_dir', default='')
py.arg('--ckpt_single_best', type=bool, default=True)
py.arg('--ckpt_max_to_keep', type=int, default=1)
py.arg('--ckpt_keep_checkpoint_every_n_hours', type=int, default=1000)
py.arg('--restore_from', default='')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--flip', type=bool, default=True)     # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--log_hidden_metrics_every', type=int, default=5000)
py.arg('--log_img_every', type=int, default=500)
py.arg('--write_n_grid_imgs', type=int, default=3)
py.arg('--write_n_row_imgs', type=int, default=20)
py.arg('--write_n_swap_imgs', type=int, default=24)
py.arg('--write_imgs_to_disk', type=bool, default=False)
py.arg('--save_img_to_tb', type=bool, default=True) 
py.arg('--gen_lr', type=float, default=0.0001)
py.arg('--dis_lr', type=float, default=0.00005)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--guess_loss_weight', type=float, default=0.0)
py.arg('--always_train_guess_disc', type=bool, default=False)
py.arg('--defence_noise_sigma', type=float, default=0.0)
py.arg('--style_noise_sigma', type=float, default=0.0)
py.arg('--style_rand_rec_weight', type=float, default=0.0)
py.arg('--style_swap_rec_weight', type=float, default=0.0)
py.arg('--cycle_swap_weight', type=float, default=0.0)
py.arg('--style_zero_rec_weight', type=float, default=0.0)
py.arg('--style_idt_rec_weight', type=float, default=0.0)
py.arg('--style_norm_weight', type=float, default=0.0)
py.arg('--gt_data_weight', type=float, default=0.0)

py.arg('--style_swap_loss_fn', default='l1', choices=['l1', 'l2', 'max'])
py.arg('--cycle_swap_loss_fn', default='l1', choices=['l1', 'l2', 'max'])

py.arg('--seed', type=int, default=0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples

args = py.args()

dataset_defaults = {
    'dshapes_split_color_vs_ori_size': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'dshapes_split_color_vs_ori_size/'),
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all_all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'shape_v2/'),
        'is_categorical_attr': {
            'f_hue': True, 'w_hue': True, 'o_hue': True, 
            'size': False, 'shape': True, 'ori': False
        },
        'eval_mode': 'gt',
        'infer_eval_attr_role': {
            'f_hue': 'A', 'w_hue': 'A', 'o_hue': 'C', 
            'size': 'B', 'shape': 'C', 'ori': 'B'
        },
        'final_gt_score_weights': {
            'f_hue': 1.0, 'w_hue': 1.0, 'o_hue': 1.0, 
            'size': 1/8.0, 'shape': 1.0, 'ori': 1/15.0
        }
    },
    'dshapes_ohue_size_vs_shape_whue': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'dshapes_ohue_size_vs_shape_whue/'),
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all_all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'shape_v2/'),
        'is_categorical_attr': {
            'f_hue': True, 'w_hue': True, 'o_hue': True, 
            'size': False, 'shape': True, 'ori': False
        },
        'eval_mode': 'gt',
        'infer_eval_attr_role': {
            'f_hue': 'C', 'w_hue': 'B', 'o_hue': 'A', 
            'size': 'A', 'shape': 'B', 'ori': 'C'
        },
        'final_gt_score_weights': {
            'f_hue': 1.0, 'w_hue': 1.0, 'o_hue': 1.0, 
            'size': 1/8.0, 'shape': 1.0, 'ori': 1/15.0
        }
    },
    'dshapes_fhue_shape_vs_ori_ohue': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'dshapes_fhue_shape_vs_ori_ohue/'),
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'shape_v2/'),
        'is_categorical_attr': {
            'f_hue': True, 'w_hue': True, 'o_hue': True, 
            'size': False, 'shape': True, 'ori': False
        },
        'eval_mode': 'gt',
        'infer_eval_attr_role': {
            'f_hue': 'A', 'w_hue': 'C', 'o_hue': 'B', 
            'size': 'C', 'shape': 'A', 'ori': 'B'
        },
        'final_gt_score_weights': {
            'f_hue': 1.0, 'w_hue': 1.0, 'o_hue': 1.0, 
            'size': 1/8.0, 'shape': 1.0, 'ori': 1/15.0
        }
    },
    'synaction_bg_vs_idt_v2': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'synaction_bg_vs_idt_v2/'),
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': ''
    },
    'synaction_bg_vs_idt_v3': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'synaction_bg_vs_idt_v3/'),
        'gt_dataset_fn': 'gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'synaction3_v3/'),
        'is_categorical_attr': {
            'idt': True, 'bg': True,
            'pose-0': False, 'pose-1': False, 'pose-2': False, 'pose-3': False, 'pose-4': False, 'pose-5': False, 'pose-6': False, 'pose-7': False, 
            'pose-8': False, 'pose-9': False, 'pose-10': False, 'pose-11': False, 'pose-12': False, 'pose-13': False, 'pose-14': False, 'pose-15': False, 
        },
        'gt_attr_clf_input_size': 64,
        'eval_mode': 'infer',
        'infer_eval_attr_role': {
            'idt': 'B', 'bg': 'A',
            'pose-0': 'C', 'pose-1': 'C', 'pose-2': 'C', 'pose-3': 'C', 'pose-4': 'C', 'pose-5': 'C', 'pose-6': 'C', 'pose-7': 'C', 
            'pose-8': 'C', 'pose-9': 'C', 'pose-10': 'C', 'pose-11': 'C', 'pose-12': 'C', 'pose-13': 'C', 'pose-14': 'C', 'pose-15': 'C', 
        },
        'attr_mean_groups' : {
            'pose': ['pose-%d' % x for x in range(16)]
        },
        'final_gt_score_weights': {
            'idt': 1.0, 'bg': 1.0, 'pose': 1.0
        }
    },
    'celeba_custom_split': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'celeba_custom_split/'),
        'gt_dataset_fn': 'pseudo_gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'celeba_v2/'),
        'is_categorical_attr': {
            'Hair_Color': True, 'Facial_Hair': True, 'Eyeglasses': True, 'Wearing_Hat': True, 
            'Young': True, 'Smiling': True, 'Male': True, 
            'ori-x': False, 'ori-y': False, 'ori-z': False,
            'bg-r': False, 'bg-g': False, 'bg-b': False, 'skin-r': False, 'skin-g': False, 'skin-b': False},
        'gt_attr_clf_input_size': 64,
        'eval_mode': 'infer',
        'infer_eval_attr_role': {
            'Hair_Color': 'B', 'Facial_Hair': 'A', 'Eyeglasses': 'C', 'Wearing_Hat': 'C', 'Young': 'A', 'Smiling': 'A', 'Male': 'D',
            'ori-x': 'C', 'ori-y': 'C', 'ori-z': 'C', 'bg-r': 'C', 'bg-g': 'C', 'bg-b': 'C', 'skin-r': 'C', 'skin-g': 'C', 'skin-b': 'C'
        },
        'attr_mean_groups' : {
            'ori': ['ori-x', 'ori-y', 'ori-z'], 'bg': ['bg-r', 'bg-g', 'bg-b'], 'skin': ['skin-r', 'skin-g', 'skin-b']
        },
        'final_gt_score_weights': {
            'Hair_Color': 1.0, 'Facial_Hair': 1.0, 'Eyeglasses': 1.0, 'Wearing_Hat': 1.0, 'Young': 1.0, 'Smiling': 1.0, 'Male': 1.0,
            'ori': 1.0, 'bg': 1.0, 'skin': 1.0,
        }
    },
    'celeba_custom_split_sw2': {
        'gt_dataset_root': os.path.join(args.gt_main_dataset_root, 'celeba_custom_split/'),
        'gt_dataset_fn': 'pseudo_gt.txt',
        'gt_dataset_img_folder': 'all',
        'gt_attr_clf_model_path': os.path.join(args.gt_main_attr_clf_model_root, 'celeba_v2/'),
        'is_categorical_attr': {
            'Hair_Color': True, 'Facial_Hair': True, 'Eyeglasses': True, 'Wearing_Hat': True, 
            'Young': True, 'Smiling': True, 'Male': True, 
            'ori-x': False, 'ori-y': False, 'ori-z': False,
            'bg-r': False, 'bg-g': False, 'bg-b': False, 'skin-r': False, 'skin-g': False, 'skin-b': False},
        'gt_attr_clf_input_size': 64,
        'eval_mode': 'infer',
        'infer_eval_attr_role': {
            'Hair_Color': 'B', 'Facial_Hair': 'A', 'Eyeglasses': 'C', 'Wearing_Hat': 'C', 'Young': 'A', 'Smiling': 'A', 'Male': 'D',
            'ori-x': 'C', 'ori-y': 'C', 'ori-z': 'C', 'bg-r': 'C', 'bg-g': 'C', 'bg-b': 'C', 'skin-r': 'C', 'skin-g': 'C', 'skin-b': 'C'
        },
        'attr_mean_groups' : {
            'ori': ['ori-x', 'ori-y', 'ori-z'], 'bg': ['bg-r', 'bg-g', 'bg-b'], 'skin': ['skin-r', 'skin-g', 'skin-b']
        },
        'final_gt_score_weights': {
            'Hair_Color': 1.0, 'Facial_Hair': 1.0, 'Eyeglasses': 0.0, 'Wearing_Hat': 0.0, 'Young': 0.0, 'Smiling': 0.0, 'Male': 1.0,
            'ori': 1.5, 'bg': 0.8, 'skin': 0.3,
        }
    }
}

to_compare_names = [('A2B_B_s', 'A', 'B', 'A2B_B_s_gt'), ('B2A_A_s', 'B', 'A', 'B2A_A_s_gt')]
role_names = ['source', 'guide', 'gt']

for k, v in dataset_defaults[args.gt_dataset_name].items():
    setattr(args, k, v)

tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loss_functions = {
    'l2': lambda x, y: tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(x-y), axis=(1, 2, 3)))),
    'l1': tf.losses.MeanAbsoluteError(),
    'max': lambda x, y: tf.reduce_mean(tf.reduce_max(tf.abs(x-y), axis=(1, 2, 3)))
}

@tf.function
def attr_clf_model_predict(imgs):
    if args.gt_attr_clf_input_size != '':
        size = args.gt_attr_clf_input_size
        imgs = tf.image.resize(imgs, (size, size))
    return attr_clf_model(imgs)

def predict_attrs(imgs, is_cat_dict):
    preds = attr_clf_model_predict(imgs)
    final_pred_dict = {
        n: (np.argmax(x, axis=1) if is_cat_dict[n] else x[:, 0]) 
        for n, x in zip(attr_clf_model.output_names, preds)
    }
    return final_pred_dict

def get_gt_attr_val(attr_preds, cmp_name, cmp_role, attr_name):
    if args.eval_mode == 'gt':
        return attr_preds[cmp_name][attr_name]
    elif args.eval_mode == 'infer':
        if cmp_name in ['A', 'B']:
            assert cmp_role in ['source', 'guide']
            return attr_preds[cmp_name][attr_name]
        else:
            assert cmp_name in ['A2B_B_s_gt', 'B2A_A_s_gt']
            assert cmp_role == 'gt'
            source_dom, guide_dom = cmp_name[0], cmp_name[2]
            attr_role = args.infer_eval_attr_role[attr_name]
            if attr_role == 'C':
                # e.g. pred['A' / 'B'][attr] if cmp_name = 'A2B_B_s_gt' / 
                return attr_preds[source_dom][attr_name]
            elif attr_role in ['D', source_dom, guide_dom]:
                return attr_preds[guide_dom][attr_name]
            else:
                raise ValueError('unexpected attr_role %s %s' % (attr_role, [source_dom, guide_dom]))
    else:
        raise ValueError('unexpected eval_mode %s', args.eval_mode)

def update_attr_metrics(metrics, output_dict, to_compare_names, role_names, is_cat_dict):
    for comp_names in to_compare_names:
        attr_preds = {n: predict_attrs(output_dict[n], is_cat_dict) for n in comp_names}
        gen_name = comp_names[0] # 'A2B_B_s'
        split_name = gen_name[:3]  # 'A2B'
        # ['source', 'guide', 'gt'], ['A', 'B', 'A2B_B_s_gt']
        other_names = dict(zip(role_names, comp_names[1:]))
        for attr_name in attr_preds[gen_name].keys():
            for cmp_role, cmp_name in other_names.items():
                y_true = get_gt_attr_val(attr_preds, cmp_name, cmp_role, attr_name)
                # y_true = attr_preds[cmp_name][attr_name]  # ['A'][attr] / ['A2B_B_s_gt'][attr]
                y_pred = attr_preds[gen_name][attr_name]  # ['A2B_B_s'][attr]
                # ['A2B']['source'][attr] / ['A2B']['gt'][attr]
                metrics[split_name][cmp_role][attr_name].update_state(y_true, y_pred)

def get_disent_fn(model):
    def fn(x, y, **kwargs):
        c = tf.concat([x, y], axis=-1)
        out = model(c, **kwargs)
        return out[..., :-disent_dim], out[..., -disent_dim:]
    return fn


def with_noise(t, a):
    if a > 0.0:
        noise = tf.random.normal(t.shape) * a
        return t + noise
    else:
        return t

# output_dir
folder_suffix = ''
for suf_i in range(1, 20):
    potential_folder = args.output_dir.rstrip("/") + folder_suffix
    if (os.path.exists(potential_folder) 
        and os.path.exists(os.path.join(potential_folder, 'settings.yml'))):
        folder_suffix = '_%d' % suf_i
    else:
        output_dir = potential_folder
        py.mkdir(output_dir)
        break
else:
    raise RuntimeError('all potential folders exist! weird!')

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)
print(args)

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

A_B_dataset_test = tf.data.Dataset.zip(tuple(datasets)).repeat()

if args.shuffle_supervised_pairs_buffer > 0:
    suffled_datasets = [
        ds.shuffle(args.shuffle_supervised_pairs_buffer, reshuffle_each_iteration=False)
        for ds in datasets]
    A_B_dataset = tf.data.Dataset.zip(tuple(suffled_datasets))
else:
    A_B_dataset = tf.data.Dataset.zip(tuple(datasets))

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

guess_dim = 6
D_A_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
D_B_guess = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, guess_dim))
extra_ckpt_dict = {'D_A_guess': D_A_guess, 'D_B_guess': D_B_guess}

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()

sty_swap_loss_fn = loss_functions[args.style_swap_loss_fn]
img_swap_loss_fn = loss_functions[args.cycle_swap_loss_fn]
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.gen_lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.dis_lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


def gen_noisy_G_fn(G_fn):
    def func(img_x, s_y):
        return G_fn(
            with_noise(img_x, args.defence_noise_sigma), 
            with_noise(s_y, args.style_noise_sigma), training=True)
    return func


G_A2B_fn_w_nz = gen_noisy_G_fn(G_A2B_fn)
G_B2A_fn_w_nz = gen_noisy_G_fn(G_B2A_fn)


@tf.function
def train_G(A, B, A2B_B_s_gt, B2A_A_s_gt):
    with tf.GradientTape() as t:
        s_zero = tf.zeros((*A.shape[:-1], 1))
        s_a = G_A2B_fn(A, s_zero, training=True)[1]
        s_b = G_B2A_fn(B, s_zero, training=True)[1]
        s_a_rand, s_b_rand = [tf.roll(x, 1, 0) for x in [s_a, s_b]]
        A2B, s_a_nz = G_A2B_fn_w_nz(A, s_b_rand)
        B2A, s_b_nz = G_B2A_fn_w_nz(B, s_a_rand)
        A2B2A, s_b_rand_rec = G_B2A_fn_w_nz(A2B, s_a)
        B2A2B, s_a_rand_rec = G_A2B_fn_w_nz(B2A, s_b)
        A2A, s_a_idt = G_B2A_fn_w_nz(A, s_a)
        B2B, s_b_idt = G_A2B_fn_w_nz(B, s_b)

        A2B_B_s, _ = G_A2B_fn_w_nz(A, s_b)
        B2A_A_s, _ = G_B2A_fn_w_nz(B, s_a)

        # should be equivalently styles s_a and s_b
        s_a_swap = G_A2B_fn(B2A_A_s, s_zero, training=True)[1]
        s_b_swap = G_B2A_fn(A2B_B_s, s_zero, training=True)[1]
        
        A2B2A_swap = G_B2A_fn_w_nz(A2B, s_a_swap)[0]
        B2A2B_swap = G_A2B_fn_w_nz(B2A, s_b_swap)[0]

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

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

        style_swap_rec_losses = [
            sty_swap_loss_fn(s_a, s_a_swap),
            sty_swap_loss_fn(s_b, s_b_swap)]

        cycle_swap_losses = [
            img_swap_loss_fn(A, A2B2A_swap), 
            img_swap_loss_fn(B, B2A2B_swap)]

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

        if args.style_swap_rec_weight > 0:
            G_losses.append((style_swap_rec_losses[0] + style_swap_rec_losses[1]) * args.style_swap_rec_weight)

        if args.cycle_swap_weight > 0:
            G_losses.append((cycle_swap_losses[0] + cycle_swap_losses[1]) * args.cycle_swap_weight)

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
        'style_norm_loss': style_norm_loss,
        'style_A_swap_rec_loss': style_swap_rec_losses[0],
        'style_B_swap_rec_loss': style_swap_rec_losses[1],
        'A_cycle_swap_loss': cycle_swap_losses[0],
        'B_cycle_swap_loss': cycle_swap_losses[1],
        'total': G_loss,
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

        A_fwd_guess_logits = D_A_guess(tf.concat([A2B2A, A], axis=-1), training=True)
        A_rev_guess_logits = D_A_guess(tf.concat([A, A2B2A], axis=-1), training=True)
        B_fwd_guess_logits = D_B_guess(tf.concat([B2A2B, B], axis=-1), training=True)
        B_rev_guess_logits = D_B_guess(tf.concat([B, B2A2B], axis=-1), training=True)

        A_rev_guess_loss, A_fwd_guess_loss = d_loss_fn(A_rev_guess_logits, A_fwd_guess_logits)
        B_rev_guess_loss, B_fwd_guess_loss = d_loss_fn(B_rev_guess_logits, B_fwd_guess_logits)

        if args.always_train_guess_disc:
            train_guess_loss_weight = 1.0
        else:
            train_guess_loss_weight = args.guess_loss_weight

        if train_guess_loss_weight:
            D_losses.append(
                (A_fwd_guess_loss + A_rev_guess_loss + B_fwd_guess_loss + B_rev_guess_loss) * train_guess_loss_weight
            )

        D_loss = sum(D_losses)

    D_vars = (
        D_A.trainable_variables + D_B.trainable_variables 
        + D_A_guess.trainable_variables + D_B_guess.trainable_variables)

    D_grad = t.gradient(D_loss, D_vars)
    D_optimizer.apply_gradients(zip(D_grad, D_vars))

    summary = {
        'A_d_loss': A_d_loss + B2A_d_loss, 
        'B_d_loss': B_d_loss + A2B_d_loss,
        'D_A_gp': D_A_gp, 'D_B_gp': D_B_gp,
        'total': D_loss
    }

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
    A2B_rand_s = G_A2B_fn_w_nz(A, s_b_rand)[0]
    B2A_rand_s = G_B2A_fn_w_nz(B, s_a_rand)[0]
    A2B2A = G_B2A_fn_w_nz(A2B_rand_s, s_a)[0]
    B2A2B = G_A2B_fn_w_nz(B2A_rand_s, s_b)[0]
    A2B2A_perm = G_B2A_fn_w_nz(A2B_rand_s, s_a_perm)[0]
    B2A2B_perm = G_A2B_fn_w_nz(B2A_rand_s, s_b_perm)[0]
    A2B2A_rand = G_B2A_fn_w_nz(A2B_rand_s, s_a_rand)[0]
    B2A2B_rand = G_A2B_fn_w_nz(B2A_rand_s, s_b_rand)[0]
    
    A2B_B_s = G_A2B_fn_w_nz(A, s_b)[0]
    B2A_A_s = G_B2A_fn_w_nz(B, s_a)[0]

    s_a_swap = G_A2B_fn(B2A_A_s, s_zero, training=True)[1]
    s_b_swap = G_B2A_fn(A2B_B_s, s_zero, training=True)[1]
    
    A2B2A_swap = G_B2A_fn_w_nz(A2B_rand_s, s_a_swap)[0]
    B2A2B_swap = G_A2B_fn_w_nz(B2A_rand_s, s_b_swap)[0]

    names = ['A2B', 'B2A', 'A2B2A', 'B2A2B', 'A_perm', 'B_perm', 
            'A2B2A_perm', 'B2A2B_perm', 'A2B2A_rand', 'B2A2B_rand',
            'A2B_B_s', 'B2A_A_s', 'A2B2A_swap', 'B2A2B_swap']

    values = [A2B_rand_s, B2A_rand_s, A2B2A, B2A2B, A_perm, B_perm, 
              A2B2A_perm, B2A2B_perm, A2B2A_rand, B2A2B_rand,
              A2B_B_s, B2A_A_s, A2B2A_swap, B2A2B_swap]
    
    outputs = dict(zip(names, values))
    return outputs


@tf.function
def sample_grid_zs(A, B):
    batch = A.shape[0]
    s_zero = tf.zeros((*A.shape[:-1], 1))
    s_a = G_A2B_fn(A, s_zero, training=True)[1]
    s_b = G_B2A_fn(B, s_zero, training=True)[1]

    A2B_B_s_grid_flat_list, B2A_A_s_grid_flat_list = [], []
    for k in range(batch):
      A2B_B_s_grid_flat_list.append(
          G_A2B_fn_w_nz(tf.repeat(A[k, None], batch, 0), s_b)[0])
      B2A_A_s_grid_flat_list.append(
          G_B2A_fn_w_nz(tf.repeat(B[k, None], batch, 0), s_a)[0]
      )
    A2B_B_s_grid_flat = tf.concat(A2B_B_s_grid_flat_list, axis=0)
    B2A_A_s_grid_flat = tf.concat(B2A_A_s_grid_flat_list, axis=0)
    A2B_B_s_grid, B2A_A_s_grid = [
      tf.reshape(x, (batch, batch, *A.shape[1:])) 
      for x in [A2B_B_s_grid_flat, B2A_A_s_grid_flat]]
    return A2B_B_s_grid, B2A_A_s_grid


def get_total_grid(A, B, A2B_B_s_grid):
  batch = A.shape[0]
  img_s = A.shape[1]
  draw_col = A.numpy().reshape(batch*img_s, img_s, 3)
  draw_row = B.numpy().swapaxes(0, 1).reshape(img_s, batch*img_s, 3)
  draw_grid = A2B_B_s_grid.numpy().swapaxes(1, 2).reshape(img_s*batch, img_s*batch, 3)
  total_grid_items = [[np.zeros((img_s, img_s, 3)), draw_row], [draw_col, draw_grid]]
  total_grid = np.concatenate([np.concatenate(x, axis=1) for x in total_grid_items], axis=0)
  return total_grid


def get_AB_zs_grids(A, B):
    A2B_B_s_grid, B2A_A_s_grid = sample_grid_zs(A, B)
    grid_A = get_total_grid(A, B, A2B_B_s_grid)
    grid_B = get_total_grid(B, A, B2A_A_s_grid)
    return grid_A, grid_B


@tf.function
def sample_grid_cyc(A, B, sub_idx):
    batch = A.shape[0]
    s_zero = tf.zeros((*A.shape[:-1], 1))
    s_a = G_A2B_fn(A, s_zero, training=True)[1]
    s_b = G_B2A_fn(B, s_zero, training=True)[1]
    s_b_rep = tf.repeat(s_b[sub_idx, None], batch, 0)
    A2B_B_first_s = G_A2B_fn_w_nz(A, s_b_rep)[0]
    s_a_rep = tf.repeat(s_a[sub_idx, None], batch, 0)
    B2A_A_first_s = G_B2A_fn_w_nz(B, s_a_rep)[0]

    A2B2A_A_s_grid_flat_list, B2A2B_B_s_grid_flat_list = [], []
    for k in range(batch):
      A2B2A_A_s_grid_flat_list.append(
          G_B2A_fn_w_nz(tf.repeat(A2B_B_first_s[k, None], batch, 0), s_a)[0])
      B2A2B_B_s_grid_flat_list.append(
          G_A2B_fn_w_nz(tf.repeat(B2A_A_first_s[k, None], batch, 0), s_b)[0])
    
    A2B2A_A_s_grid_flat = tf.concat(A2B2A_A_s_grid_flat_list, axis=0)
    B2A2B_B_s_grid_flat = tf.concat(B2A2B_B_s_grid_flat_list, axis=0)
    A2B2A_A_s_grid, B2A2B_B_s_grid = [
      tf.reshape(x, (batch, batch, *A.shape[1:])) 
      for x in [A2B2A_A_s_grid_flat, B2A2B_B_s_grid_flat]]
    return A2B2A_A_s_grid, B2A2B_B_s_grid


def get_total_cyc_grid(A, A2B2A_A_s_grid):
  batch = A.shape[0]
  img_s = A.shape[1]
  draw_col = A.numpy().reshape(batch*img_s, img_s, 3)
  draw_row = A.numpy().swapaxes(0, 1).reshape(img_s, batch*img_s, 3)
  draw_grid = A2B2A_A_s_grid.numpy().swapaxes(1, 2).reshape(img_s*batch, img_s*batch, 3)
  total_grid_items = [[np.zeros((img_s, img_s, 3)), draw_row], [draw_col, draw_grid]]
  total_grid = np.concatenate([np.concatenate(x, axis=1) for x in total_grid_items], axis=0)
  return total_grid

def get_AB_cyc_grids(A, B, sub_idx):
    A2B2A_A_s_grid, B2A2B_B_s_grid = sample_grid_cyc(A, B, sub_idx)
    grid_A = get_total_cyc_grid(A, A2B2A_A_s_grid)
    grid_B = get_total_cyc_grid(B, B2A2B_B_s_grid)
    return grid_A, grid_B

# /// hidden eval

tfk = tf.keras
tfkl = tf.keras.layers

def predict_attrs_tf(imgs, is_cat_dict):
    preds = attr_clf_model_predict(imgs)
    final_pred_dict = {
        n: (tf.math.argmax(x, axis=1) if is_cat_dict[n] else x[:, 0]) 
        for n, x in zip(attr_clf_model.output_names, preds)
    }
    return final_pred_dict

def gen_fn_hidden_attr_dataset(A, B, A2B_B_s_gt, B2A_A_s_gt):
  img_preds = sample_zs(A, B)
  output = []
  for key in ['A2B_B_s', 'B2A_A_s']:
    pred_img = img_preds[key]
    gt = {}
    for input_name, input_imgs in zip(['A', 'B'], [A, B]):
      preds = predict_attrs_tf(input_imgs, args.is_categorical_attr)
      for attr_name, attr_val in preds.items():
        gt[input_name + '_' + attr_name] = attr_val
    output.append((pred_img, gt))
  return tuple(output)


def get_hidden_metric_groups():
    attr_metric_names = {}
    for attr_name, is_cat in args.is_categorical_attr.items():
        attr_metric_names[attr_name] = 'acc' if is_cat else 'mae'

    assert all(len(x) == 3 for x in set(attr_metric_names.values()))

    directed_metric_groups = {}
    for eval_direction in ['A2B', 'B2A']:
        source_dir = eval_direction[0]
        target_dir = eval_direction[2]
        other_domain = {'A': 'B', 'B': 'A'}
        metric_groups = {
            'fixed': [], 'used-sofs': [], 'used-cofc': [], 
            'dropped-cofs': [], 'dropped-sofc': []
        }

        for attr_name, attr_domain in args.infer_eval_attr_role.items():
            if attr_domain in ['A', 'B']:
                fixed_names = (other_domain[attr_domain], attr_name, attr_metric_names[attr_name])
                metric_groups['fixed'].append('_'.join(fixed_names))

                add_to = 'used-sofs' if attr_domain == target_dir else 'dropped-sofc'
                used_sofs = (attr_domain, attr_name, attr_metric_names[attr_name])
                metric_groups[add_to].append('_'.join(used_sofs))
            elif attr_domain == 'C':
                used_cofc = (source_dir, attr_name, attr_metric_names[attr_name])
                metric_groups['used-cofc'].append('_'.join(used_cofc))

                dropped_cofs = (target_dir, attr_name, attr_metric_names[attr_name])
                metric_groups['dropped-cofs'].append('_'.join(dropped_cofs))
            else:
                assert attr_domain == 'D'
        
        directed_metric_groups[eval_direction] = metric_groups

    return directed_metric_groups

def log_hidden_metrics():    
  cache_file = py.join(output_dir, 'hidden_cache')
  if os.path.exists(cache_file+'.index'):
      os.remove(cache_file+'.data-00000-of-00001')
      os.remove(cache_file+'.index')

  hidden_attr_ds = A_B_dataset.map(gen_fn_hidden_attr_dataset)
  hidden_attr_ds_cached = hidden_attr_ds.cache(cache_file)
  print(len(list(tqdm.tqdm(hidden_attr_ds_cached, desc='Generating a Dataset for Hidden Attr Detection'))))
  print(len(list(tqdm.tqdm(hidden_attr_ds_cached))))

  for idx, (eval_direction, hid_model) in enumerate(zip(['A2B', 'B2A'], hidden_models)):
    hid_clf_ds = (
        hidden_attr_ds_cached.map(lambda *x: x[idx])
                              .unbatch().batch(512))

    init_loss = hid_model.evaluate(hid_clf_ds.take(3))[0]

    hid_model.fit(
        x=hid_clf_ds.skip(3),
        validation_data=hid_clf_ds.take(3),
        epochs=300,
        callbacks=[tfk.callbacks.EarlyStopping(
            patience=30, min_delta=init_loss*0.01, restore_best_weights=True)]
    )

    hidden_metrics = dict(zip(hid_model.metrics_names, hid_model.evaluate(hid_clf_ds.take(3))))
    hidden_metrics = {k: v for k, v in hidden_metrics.items() if not k.endswith('loss')}

    grouped_hidden_metrics = {
        group_k: {metric_name: hidden_metrics[metric_name] for metric_name in group_metrics}
        for group_k, group_metrics in hidden_metric_groups[eval_direction].items()
    }

    if hasattr(args, 'attr_mean_groups'):
      final_grouped_hidden_metrics = {}
      for group_k, hidden_metrics in grouped_hidden_metrics.items():
        meaned_hidden_metrics = {}
        for name, value in hidden_metrics.items():
          for mean_group_name, mean_group_members in args.attr_mean_groups.items():
            just_dom, metric_less_name, just_metric = name[:2], name[2:-4], name[-4:]
            if metric_less_name in mean_group_members:
              full_new_name = just_dom + mean_group_name + just_metric
              if full_new_name not in meaned_hidden_metrics:
                meaned_hidden_metrics[full_new_name] = 0.0
              meaned_hidden_metrics[full_new_name] += value
              break
          else:
            meaned_hidden_metrics[name] = value
        final_grouped_hidden_metrics[group_k] = meaned_hidden_metrics
    else:
      final_grouped_hidden_metrics = grouped_hidden_metrics

    for group_name, grouped_metrics in final_grouped_hidden_metrics.items():
      for metric_name, metric_value in grouped_metrics.items():
        key = 'hidden_attr/%s-%s-%s' % (eval_direction, group_name, metric_name)
        tf.summary.scalar(key, metric_value, step=G_optimizer.iterations)
        print(key, metric_value)

  os.remove(cache_file+'.data-00000-of-00001')
  os.remove(cache_file+'.index')


def get_hidden_prediction_model(encoder, prediction_specs, per_head_sizes):
  input = tfkl.Input(shape=encoder.input_shape[1:])
  enc = encoder(input)
  outputs = []
  for attr_name, outputs_k, is_categorical in prediction_specs:
    for input_name in ['A', 'B']:
      output_name = input_name + '_' + attr_name
      x = enc
      for head_layer_size in per_head_sizes:
        x = tfkl.Dense(head_layer_size, activation='relu')(x)
        x = tfkl.Dropout(0.1)(x)

      if is_categorical:
        x = tfkl.Dense(outputs_k, activation='softmax', name=output_name)(x)
      else:
        x = tfkl.Dense(1, activation=None, name=output_name)(x)

      outputs.append(x)
  
  return tfk.Model(input, outputs)


def get_compiled_hidden_attr_model():
  test_output = attr_clf_model(tf.zeros((1, args.gt_attr_clf_input_size, args.gt_attr_clf_input_size, 3)))
  output_dims = {name: arr.shape[1] for arr, name in zip(test_output, attr_clf_model.output_names)}
  prediction_specs = [(n, output_dims[n], is_cat) for n, is_cat in args.is_categorical_attr.items()]

  encoder = tfk.Sequential([
    tfkl.Conv2D(64, 3, activation='relu', input_shape=(64, 64, 3)),
    tfkl.AveragePooling2D(2),
    tfkl.Conv2D(64, 3, activation='relu'),
    tfkl.AveragePooling2D(2),
    tfkl.Conv2D(128, 3, activation='relu'),
    tfkl.Conv2D(256, 3, activation='relu'),
    tfkl.GlobalAveragePooling2D()
  ])

  model = get_hidden_prediction_model(encoder, prediction_specs, (128,))

  loss_fns, metric_fns = {}, {}
  for input_name in ['A', 'B']:
    for attr_name, size, is_cat in prediction_specs:
      output_name = input_name + '_' + attr_name
      loss_fn_name = 'sparse_categorical_crossentropy' if is_cat else 'mse'
      loss_fns[output_name] = loss_fn_name
      metric_fns[output_name] = 'acc' if is_cat else 'mae'

  model.compile(optimizer='adam', loss=loss_fns, metrics=metric_fns)
  return model

hidden_metric_groups = get_hidden_metric_groups()
hidden_models = [get_compiled_hidden_attr_model(), get_compiled_hidden_attr_model()]

# /// hidden eval

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
ckpt_dict = dict(G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A, D_B=D_B, 
                 G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                 ep_cnt=ep_cnt, **extra_ckpt_dict)

checkpoint = tl.Checkpoint(ckpt_dict,
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=max(args.ckpt_max_to_keep, 1),
                           keep_checkpoint_every_n_hours=args.ckpt_keep_checkpoint_every_n_hours)
try:
    if args.restore_from != '':
        ckpt_fn = args.restore_from
    else:
        ckpt_fn = None

    checkpoint.restore(ckpt_fn).assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries'))

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# otherwise might have issues with swapping / random
assert args.batch_size > 3


def get_metric_dict():
    result = {}
    for attr_n, is_cat in args.is_categorical_attr.items():
        result[attr_n] = (
            tf.keras.metrics.Accuracy() if is_cat 
            else tf.keras.metrics.MeanAbsoluteError())

    return result

assert args.write_n_grid_imgs * args.batch_size > args.write_n_row_imgs

best_total_err_value = float('inf')

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

            if args.shuffle_supervised_pairs_buffer > 0:
                del G_loss_dict['gt_data_loss']

            # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            iter_i = G_optimizer.iterations.numpy()

            if (args.log_hidden_metrics_every > 0 and iter_i > 0 and iter_i % args.log_hidden_metrics_every == 0):
                log_hidden_metrics()

            # sample
            if iter_i % args.log_img_every == 0:
                if args.gt_attr_clf_model_path != '':
                    metrics = {
                        split: {rn: get_metric_dict() for rn in role_names} for split in ['A2B', 'B2A']
                    }

                gt_losses = [[], []]

                imgs = []
                swap_imgs = []
                grid_imgs = []
                grid_cyc_imgs = []

                test_iter = iter(A_B_dataset_test)
                for img_idx in range(args.write_n_grid_imgs):
                    A, B, A2B_B_s_gt, B2A_A_s_gt = next(test_iter)
                    output_sample_dict = sample_zs(A, B)
                    output_sample_dict.update(
                        dict(zip(['A', 'B', 'A2B_B_s_gt', 'B2A_A_s_gt'], [A, B, A2B_B_s_gt, B2A_A_s_gt])))
                    img_names_to_merge = [
                        'A', 'A2B', 'A2B2A', 'A2B2A_rand', 'A_perm', 'A2B2A_perm', 'A2B_B_s', 'A2B_B_s_gt', 'A2B2A_swap',
                        'B', 'B2A', 'B2A2B', 'B2A2B_rand', 'B_perm', 'B2A2B_perm', 'B2A_A_s', 'B2A_A_s_gt', 'B2A2B_swap']
                    img_swap_names = ['A', 'A2B_B_s', 'A2B2A_swap', 'B', 'B2A_A_s', 'B2A2B_swap']

                    for (pred_img_name, gt_img_name), gt_loss_list in zip([['A2B_B_s', 'A2B_B_s_gt'], ['B2A_A_s', 'B2A_A_s_gt']], gt_losses):
                        gt_loss_list.append(cycle_loss_fn(output_sample_dict[pred_img_name], output_sample_dict[gt_img_name]))

                    for k in range(args.write_n_row_imgs // args.write_n_grid_imgs + 1):
                        single_img_to_merge = [output_sample_dict[n][k, None] for n in img_names_to_merge]
                        img = im.immerge(np.concatenate(single_img_to_merge, axis=0), n_rows=2)
                        imgs.append((img + 1) * 0.5)

                    for k in range(args.batch_size):
                        single_img_to_merge = [output_sample_dict[n][k, None] for n in img_swap_names]
                        img = im.immerge(np.concatenate(single_img_to_merge, axis=0), n_rows=1)
                        swap_imgs.append((img + 1) * 0.5)

                    if args.gt_attr_clf_model_path != '':
                        update_attr_metrics(metrics, output_sample_dict, to_compare_names, role_names, args.is_categorical_attr)

                    grid_imgs.append(get_AB_zs_grids(A, B))
                    grid_cyc_imgs.append(get_AB_cyc_grids(A, B, 0))

                    if args.write_imgs_to_disk:
                        fn = 'iter-%09d-%d.jpg' % (iter_i, img_idx)
                        im.imwrite(img, py.join(sample_dir, fn))

                for name, gt_loss_list in zip(['A2B', 'B2A'], gt_losses):
                    tf.summary.scalar('test_gt/%s' % name, tf.reduce_mean(gt_loss_list), step=G_optimizer.iterations)

                if args.save_img_to_tb:
                    tf.summary.image(
                        'img', imgs, 
                        step=G_optimizer.iterations, max_outputs=args.write_n_row_imgs)

                    if grid_imgs:
                        for name, grid in zip(['A', 'B'], zip(*grid_imgs)):
                            tf.summary.image('grid-%s' % name, 
                                [x * 0.5 + 0.5 for x in grid], 
                                step=G_optimizer.iterations,
                                max_outputs=args.write_n_grid_imgs)

                    if grid_cyc_imgs:
                        for name, grid in zip(['A', 'B'], zip(*grid_cyc_imgs)):
                            tf.summary.image('grid-cyc-%s' % name, 
                                [x * 0.5 + 0.5 for x in grid], 
                                step=G_optimizer.iterations,
                                max_outputs=args.write_n_grid_imgs)


                if args.gt_attr_clf_model_path != '':
                    metric_values = {}
                    for split_name, split_metrics in metrics.items():
                        for cmp_role, attr_metrics in split_metrics.items():
                            for attr_name, metric in attr_metrics.items():
                                if isinstance(metric, tf.keras.metrics.Accuracy):
                                    value = 1.0 - metric.result()
                                else:
                                    value = metric.result()
                                metric_values[(split_name, cmp_role, attr_name)] = value                                

                    if hasattr(args, 'attr_mean_groups'):
                        final_metric_values = {}
                        mean_group_values = defaultdict(list)                             
                        for key, value in metric_values.items():
                            split_name, cmp_role, attr_name = key
                            for mean_group_name, mean_group_attrs in args.attr_mean_groups.items():
                                if attr_name in mean_group_attrs:
                                    mean_group_values[(split_name, cmp_role, mean_group_name)].append(value)
                                    break
                            else:
                                final_metric_values[key] = value

                        for key, values in mean_group_values.items():
                            final_metric_values[key] = tf.reduce_sum(values)
                        
                        metric_values = final_metric_values

                    total_gt_errors = []
                    for (split_name, cmp_role, attr_name), value in metric_values.items():    
                        name = 'attr/%s-%s-%s' % (split_name, cmp_role, attr_name)
                        tf.summary.scalar(name, value, step=G_optimizer.iterations)

                        if cmp_role == 'gt' and hasattr(args, 'final_gt_score_weights'):
                            total_gt_errors.append(value * args.final_gt_score_weights[attr_name])                            

                    total_err_value = tf.reduce_sum(total_gt_errors)
                    tf.summary.scalar('attr/total', total_err_value, step=G_optimizer.iterations)

                    if total_err_value < best_total_err_value:
                        best_total_err_value = total_err_value
                        if args.ckpt_single_best:
                            checkpoint.checkpoint.write(py.join(output_dir, 'best/ckpt'))
                        else:
                            checkpoint.save(py.join(output_dir, 'best/ckpt'))

                        if grid_imgs:
                            for name, grid in zip(['A', 'B'], zip(*grid_imgs)):
                                tf.summary.image('best/grid-%s' % name, 
                                    [x * 0.5 + 0.5 for x in grid], 
                                    step=G_optimizer.iterations,
                                    max_outputs=args.write_n_grid_imgs)

                if args.write_n_swap_imgs > 0:
                    final_swap_img = tf.concat(swap_imgs[:args.write_n_swap_imgs], axis=0)
                    tf.summary.image('swaps', [final_swap_img], step=G_optimizer.iterations, max_outputs=1)

        # save checkpoint
        if args.ckpt_max_to_keep > 0:
            checkpoint.save(ep)