import functools
import glob

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


py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--dataset_globs', default='')
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
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = args.output_dir
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

if args.dataset == 'dshapes_split_color_vs_ori_size':
    args.dataset_globs = (
        '/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/trainA/*.png'
        ':/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/trainB/*.png'
        ':/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/testA/*.png'
        ':/scratch3/data/guess_disent/dshapes_split_color_vs_ori_size/testB/*.png')

if args.dataset_globs != '':
    img_paths = [glob.glob(x) for x in args.dataset_globs.split(':')]
    A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test = img_paths
else:
    A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
    B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
    A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
    B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')

A_B_dataset, len_dataset = data.make_zip_dataset(
    A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, 
    training=True, repeat=False, flip=args.flip)

A_B_dataset_test, _ = data.make_zip_dataset(
    A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, 
    training=False, repeat=True, flip=args.flip)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

disent_dim = 1
io_dim = 3 + disent_dim
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, io_dim), output_channels=io_dim)
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, io_dim), output_channels=io_dim)

def get_disent_fn(model):
    def fn(x, y, **kwargs):
        c = tf.concat([x, y], axis=-1)
        out = model(c, **kwargs)
        return out[..., :-disent_dim], out[..., -disent_dim:]
    return fn

G_A2B_fn = get_disent_fn(G_A2B)
G_B2A_fn = get_disent_fn(G_B2A)

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


def with_noise(t):
    if args.defence_noise_sigma > 0.0:
        noise = tf.random.normal(t.shape) * args.defence_noise_sigma
        return t + noise
    else:
        return t


@tf.function
def train_G(A, B):
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

        G_losses = [
            (A2B_g_loss + B2A_g_loss),
            (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight,
            (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight
        ]

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

    G_vars = G_A2B.trainable_variables + G_B2A.trainable_variables
    G_grad = t.gradient(G_loss, G_vars)
    G_optimizer.apply_gradients(zip(G_grad, G_vars))
    summary = {
        'A2B_g_loss': A2B_g_loss, 'B2A_g_loss': B2A_g_loss, 
        'A2B2A_cycle_loss': A2B2A_cycle_loss, 'B2A2B_cycle_loss': B2A2B_cycle_loss, 
        'A2A_id_loss': A2A_id_loss, 'B2B_id_loss': B2B_id_loss}
    
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


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

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
    return A2B_rand_s, B2A_rand_s, A2B2A, B2A2B, A_perm, B_perm, A2B2A_perm, B2A2B_perm, A2B2A_rand, B2A2B_rand


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
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

# main loop
with train_summary_writer.as_default():
    with open(py.join(output_dir, 'settings.yml')) as f:
        tf.summary.text('args', ''.join(['    ' + x for x in f.readlines()]), step=0)

    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % args.log_img_every == 0:
                imgs = []
                for img_idx in range(args.write_n_imgs):
                    A, B = next(test_iter)
                    A2B, B2A, A2B2A, B2A2B, A_perm, B_perm, A2B2A_perm, B2A2B_perm, A2B2A_rand, B2A2B_rand = sample(A, B)
                    img_to_merge = [A, A2B, A2B2A, A2B2A_rand, A_perm, A2B2A_perm, B, B2A, B2A2B, B2A2B_rand, B_perm, B2A2B_perm]
                    single_img_to_merge = [x[0, None] for x in img_to_merge]
                    img = im.immerge(np.concatenate(single_img_to_merge, axis=0), n_rows=2)
                    imgs.append((img + 1) * 0.5)

                    if args.write_imgs_to_disk:
                        fn = 'iter-%09d-%d.jpg' % (G_optimizer.iterations.numpy(), img_idx)
                        im.imwrite(img, py.join(sample_dir, fn))
                    
                if args.save_img_to_tb:
                    tf.summary.image('img', imgs, step=G_optimizer.iterations, max_outputs=args.write_n_imgs)

        # save checkpoint
        checkpoint.save(ep)
