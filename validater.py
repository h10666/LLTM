import time
import torch
import ops.utils as utils
import numpy as np
# from CLIP import clip

from opts import parser
from ops.utils import AverageMeter, accuracy
from train_helper import loss_boxes, build_box_targets, multitask_accuracy
from train_helper import create_logits, gen_label
from train_helper import save_results
from archs.Text_Prompt import *


args = parser.parse_args()
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def validate(val_loader, model, criterion, KLcriterion, epoch, matcher = None, 
        tokenizer =None, text_dict = None, classes = None, log=None, tf_writer=None, class_to_idx = None):
    test_model = True
    
    if 'ZSL' in args.split_mode:
            class_names = utils.get_class_names_ZSL(args.split_mode, 'val')
            # print(class_names)

            with open('dataset/STHELSE/category.txt') as f: ### change it
                lines = f.readlines()
                category_list = [item.split('\n')[0] for item in lines]
    else:
        class_names = []
        with open('dataset/STHELSE/category.txt') as f:
            lines = f.readlines()
            for line in lines:
                class_names.append(line.split('\n')[0]) 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if 'epic55' in args.dataset:
        if args.joint and args.do_KL:
            KL_losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        verb_losses = AverageMeter()
        # noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        # noun_top1 = AverageMeter()
        # noun_top5 = AverageMeter()
    else:
        # bp_loss = AverageMeter()
        loss1 = AverageMeter()
        loss2 = AverageMeter()
        video_top1 = AverageMeter()
        video_top5 = AverageMeter()
        fusion_top1 = AverageMeter()
        fusion_top5 = AverageMeter()
    
    logits_matrix, targets_list, preds, targets = [], [], [], []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, video_target, box_tensors, box_categories, captions, label_captions) in enumerate(val_loader):
            # print('input:\t', input.size())
            # exit(0)
            # video_target = video_target["verb"]
            data_time.update(time.time() - end)
            
            if 'ZSL' in args.split_mode:
                video_target = utils.modify_target(video_target, category_list, class_names)

            # if is_cuda:
            #     video_target = video_target.cuda()
                # video_label = video_label.cuda()

            # input_var = torch.autograd.Variable(input)
            # video_target_var = torch.autograd.Variable(video_target)
            # box_tensors_var = torch.autograd.Variable(box_tensors)
            # box_categories_var = torch.autograd.Variable(box_categories)
            input = input.to(device)
            target = video_target.to(device)
            if args.arch=='resnet50':
                pass
            else:
                box_tensors = box_tensors.to(device)
            # box_tensors = box_tensors.to(device)
                box_categories = box_categories.to(device)
            # compute output
            # output = model(global_img_tensors, box_tensors, box_categories, video_label)
            if 'epic55' in args.dataset:
                # target = {k: v.to(device) for k, v in video_target.items()}

                if args.joint:
                    num_text_aug = 2
                    text_id = np.random.randint(num_text_aug,size=len(target))
                    # print(len(captions))
                    do_KL = False
                    text_input = tags_prompt(captions, text_id, tokenizer, args.do_prompt, do_KL)
                    if args.do_KL:
                        do_KL = True
                        label_input = tags_prompt(label_captions, text_id, tokenizer, args.do_prompt, do_KL)
                    else:
                        label_input = None
                    if args.do_KL:
                        class_input=[]
                        output, tags_feas, labels_feas, logit_scale = model(input, box_tensors, box_categories, text_input, label_input, class_input, test_model)
                        # if isinstance(logit_scale, list):
                        if len(logit_scale)>1:
                            logit_scale = logit_scale[0]
                        logits_per_tag, logits_per_label = create_logits(tags_feas, labels_feas, logit_scale)
                        ground_truth = torch.tensor(gen_label(target), dtype=tags_feas.dtype,device=device)
                        kl_loss = (KLcriterion(logits_per_tag,ground_truth) + KLcriterion(logits_per_label,ground_truth))/2
                    else:
                        output = model(input_var, box_tensors_var, box_categories_var, text_input, label_input, class_input, test_model)
                else:
                    text_input=[]
                    label_input=[]
                    class_input=[]
                    output = model(input, box_tensors, box_categories, text_input, label_input, class_input, test_model)

                
                # loss_verb = criterion(output[0], target['verb'])
                loss_verb = criterion(output, target)

                # loss_noun = criterion(output[1], target['noun'])
                # loss = 0.5*(loss_verb+loss_noun)
                # loss = (1-args.delta)*loss_verb + args.delta*loss_noun
                # loss = 0.6*loss_verb + 0.4*loss_noun

                # loss = loss_verb
                if args.joint and args.do_KL:
                    KL_losses.update(kl_loss.item(), input.size(0))
                    loss=(1-args.delta)*loss_verb + args.delta*kl_loss
                else:
                    loss = loss_verb
                    
                # print(loss)
                # exit(0)
                verb_losses.update(loss_verb.item(), input.size(0))
                # noun_losses.update(loss_noun.item(), input.size(0))
                # verb_output = output[0]
                # noun_output = output[1]
                verb_prec1, verb_prec5 = accuracy(output, target, topk=(1, 5))
                verb_top1.update(verb_prec1, input.size(0))
                verb_top5.update(verb_prec5, input.size(0))
                # noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
                # noun_top1.update(noun_prec1, input.size(0))
                # noun_top5.update(noun_prec5, input.size(0))

                # prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                #                                 (target['verb'], target['noun']),
                #                                 topk=(1, 5))

                losses.update(loss.item(), input.size(0))
                # top1.update(prec1, input.size(0))
                # top5.update(prec5, input.size(0))
            ########## constractive loss calculating ###########
            elif args.arch == 'region_coord_tsm':
                text_input = None
                label_input = None
                class_input = None
                test_model = None
                output = model(input_var, box_tensors_var, box_categories_var, text_input, label_input, class_input, test_model)
                global_loss = criterion(output, video_target_var) 
            
            elif args.two_stream:
                if args.text_select == 'label_text':
                    num_text_aug = 16
                    text_id = np.random.randint(num_text_aug,size=len(video_target))
                    text_input = torch.stack([text_dict[j][i,:] for i,j in zip(video_target,text_id)])
                    text_input = text_input.to(device)
                else:
                    num_text_aug = 16
                    text_id = np.random.randint(num_text_aug,size=len(video_target))
                    text_input = captions_prompt(captions, text_id)
                    text_input = text_input.to(device)
                class_input = classes.to(device)       
                output, text_features, video_features, class_features, logit_scale = model(input_var, text_input, class_input)
                # print(video_features.size(), text_features.size(), logit_scale)
                ####################################
                # if len(logit_scale)>1:
                #         logit_scale = logit_scale[0]
                ####################################
                # print(logit_scale)
                logits_per_video, logits_per_text = create_logits(video_features,text_features,logit_scale)           
                ########################## select what loss used here #######################
                if args.loss_select == 'cost':
                    ground_truth = torch.arange(text_input.size(0),dtype=torch.long,device=device)
                    cost_loss = (criterion(logits_per_video,ground_truth) + criterion(logits_per_text,ground_truth))/2
                    # print('cost_loss:\t', cost_loss)
                elif args.loss_select == 'KL':
                    ground_truth = torch.tensor(gen_label(video_target),dtype=video_features.dtype,device=device)
                    cost_loss = (KLcriterion(logits_per_video,ground_truth) + KLcriterion(logits_per_text,ground_truth))/2
                    # print('KL_loss:\t', cost_loss)
                ##############################################################################       
                global_loss = criterion(output, video_target_var)
                # all_loss = (global_loss + cost_loss)/2
                all_loss = global_loss + cost_loss
            else:
                if args.joint:
                    num_text_aug = 2
                    text_id = np.random.randint(num_text_aug,size=len(video_target))
                    do_KL = False
                    text_input = tags_prompt(captions, text_id, tokenizer, args.do_prompt, do_KL)
                    if args.do_KL:
                        do_KL = True
                        label_input = tags_prompt(label_captions, text_id, tokenizer, args.do_prompt, do_KL)
                    else:
                        label_input = None
                else:
                    text_input = None
                    label_input = None
                class_input = None

                if args.do_KL:
                    # output, tags_feas, labels_feas, logit_scale = model(input_var, box_tensors_var, box_categories_var, text_input, label_input, class_input, test_model)
                    output, tags_feas, labels_feas, logit_scale = model(input, box_tensors, box_categories, text_input, label_input, class_input, test_model)
                    ###################################################
                    if len(logit_scale)>1:
                        logit_scale = logit_scale[0]
                    logits_per_tag, logits_per_label = create_logits(tags_feas, labels_feas, logit_scale)
                    ground_truth = torch.tensor(gen_label(video_target), dtype=tags_feas.dtype,device=device)
                    if args.do_only_c2a:
                        kl_loss = KLcriterion(logits_per_tag,ground_truth)
                    elif args.do_only_a2c:
                        kl_loss = KLcriterion(logits_per_label,ground_truth)
                    else:
                        kl_loss = (KLcriterion(logits_per_tag,ground_truth) + KLcriterion(logits_per_label,ground_truth))/2
                    ###################################################
                else:
                    # output = model(input_var, box_tensors_var, box_categories_var, text_input, label_input, class_input, test_model)
                    output = model(input, box_tensors, box_categories, text_input, label_input, class_input, test_model)
                global_loss = criterion(output, target)
                if args.do_KL:
                    # total_loss = global_loss + kl_loss
                    total_loss = (1-args.delta)*global_loss + args.delta*kl_loss
                # print('output:\t', output.size())
                # print('video_target_var:\t', video_target_var.size())
                # print('global_loss:\t', global_loss)
                # print('kl_loss:\t', kl_loss)
                # print('total_loss:\t', total_loss)
            # ############################## compute score ##############################
            if args.two_stream:
                visual_features = video_features/video_features.norm(dim=-1, keepdim=True)
                class_features /= class_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * visual_features @ class_features.T)
                similarity = similarity.view(visual_features.size(0), num_text_aug, -1).softmax(dim=-1)
                similarity = similarity.mean(dim=1, keepdim=False) 
                scores = ((output+similarity)/2)
                fusion_prec1, fusion_prec5 = accuracy(scores.data, video_target, topk=(1, 5))
            # # print('output:\t', output)
            # # print('video_target:\t', video_target)
            # # exit(0)
            video_prec1, video_prec5 = accuracy(output.data, target, topk=(1, 5))
            # ######################################################################################
            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_target.cpu().numpy())   
            # ######################################################################################
            loss1.update(global_loss.item(), input.size(0))
            
            if args.two_stream:
                losses.update(all_loss.item(), input.size(0))
                loss2.update(cost_loss.item(), input.size(0))
                # bp_loss.update(box_loss.item(), input.size(0))
                fusion_top1.update(fusion_prec1.item(), input.size(0))
                fusion_top5.update(fusion_prec5.item(), input.size(0))
            elif args.do_KL:
                losses.update(total_loss.item(), input.size(0))
                loss2.update(kl_loss.item(), input.size(0))
            else:
                pass
                        
            video_top1.update(video_prec1.item(), input.size(0))
            video_top5.update(video_prec5.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.debug == 'yes':
                break
        if 'epic55' in args.dataset:
            if args.joint and args.do_KL:
                output = ('Epoch[{0}](Train):\t'
                        'Time {batch_time.sum:.3f}\t'
                        'Data {data_time.sum:.3f}\t'
                        'Verb_loss {verb_loss.avg:.4f}\t'
                        'KL_loss {kl_loss.avg:.4f}\t'
                        'Total_loss {total_loss.avg:.4f}\t'
                        'Total_Acc1 {t_top1.avg:.1f}\t'
                        'Total_Acc5 {t_top5.avg:.1f}\t'.format(epoch, batch_time=batch_time, data_time=data_time, 
                                                            verb_loss=verb_losses, kl_loss=KL_losses, total_loss=losses, 
                                                            t_top1=verb_top1, t_top5=verb_top5))
            else:
                output = ('Epoch[{0}](Train):\t'
                    'Time {batch_time.sum:.3f}\t'
                    'Data {data_time.sum:.3f}\t'
                    'Verb_loss {verb_loss.avg:.4f}\t'
                    'Total_loss {total_loss.avg:.4f}\t'
                    'Total_Acc1 {t_top1.avg:.1f}\t'
                    'Total_Acc5 {t_top5.avg:.1f}\t'.format(epoch, batch_time=batch_time, data_time=data_time, 
                                                        verb_loss=verb_losses,total_loss=losses,
                                                        t_top1=verb_top1, t_top5=verb_top5))
        elif args.two_stream:
            output = ('Epoch[{0}](Test):\t'
                    'Time {batch_time.sum:.3f}\t'
                    'Data {data_time.sum:.3f}\t'
                    'All_Loss {loss.avg:.4f}\t'
                    'Global_loss {global_loss.avg:.4f}\t'
                    'Cost_loss {Cost_loss.avg:.4f}\t'
                    'Video_Acc1 {v_top1.avg:.1f}\t'
                    'Video_Acc5 {v_top5.avg:.1f}\t'
                    'Fusion_Acc1 {f_top1.avg:.1f}\t'
                    'Fusion_Acc5 {f_top5.avg:.1f}'.format(epoch, batch_time=batch_time, data_time=data_time, 
                                                        loss=losses, global_loss=loss1, Cost_loss=loss2,
                                                        v_top1=video_top1, v_top5=video_top5,
                                                        f_top1=fusion_top1, f_top5=fusion_top5))
        elif args.do_KL:
            output = ('Epoch[{0}](Train):\t'
                    'Time {batch_time.sum:.3f}\t'
                    'Data {data_time.sum:.3f}\t'
                    'Total_Loss {loss.avg:.4f}\t'
                    'Global_loss {global_loss.avg:.4f}\t'
                    'KL_loss {KL_loss.avg:.4f}\t'
                    'Video_Acc1 {v_top1.avg:.1f}\t'
                    'Video_Acc5 {v_top5.avg:.1f}'.format(epoch, batch_time=batch_time, data_time=data_time, 
                                                        loss=losses, global_loss=loss1, KL_loss=loss2,
                                                        v_top1=video_top1, v_top5=video_top5))
        else:     
            output = ('Epoch[{0}](Test):\t'
                    'Time {batch_time.sum:.3f}\t'
                    'Data {data_time.sum:.3f}\t'
                    'Global_loss {global_loss.avg:.4f}\t'
                    'Video_Acc1 {v_top1.avg:.1f}\t'
                    'Video_Acc5 {v_top5.avg:.1f}'.format(epoch, batch_time=batch_time, data_time=data_time, 
                                                        global_loss=loss1,
                                                        v_top1=video_top1, v_top5=video_top5))
        print(output)
        print('#'*80)
        
        if log is not None:
            log.write(output + '\n')
            log.flush()
            
        #####################################################################
        if args.evaluate:
            logits_matrix = np.concatenate(logits_matrix)
            targets_list = np.concatenate(targets_list)
            save_results(logits_matrix, targets_list, class_to_idx, args)
        #####################################################################

    if tf_writer is not None:
        if 'epic55' in args.dataset:
            if args.joint and args.do_KL:
                tf_writer.add_scalar('kl_loss/test', KL_losses.avg, epoch)
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('v_loss/test', verb_losses.avg, epoch)
            # tf_writer.add_scalar('n_loss/test', noun_losses.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top1', verb_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top5', verb_top5.avg, epoch)
            # tf_writer.add_scalar('acc/test_n_top1', noun_top1.avg, epoch)
            # tf_writer.add_scalar('acc/test_n_top5', noun_top5.avg, epoch)
            # tf_writer.add_scalar('acc/test_total_top1', top1.avg, epoch)
            # tf_writer.add_scalar('acc/test_total_top5', top5.avg, epoch)
        elif args.two_stream:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('g_loss/test', loss1.avg, epoch)
            tf_writer.add_scalar('c_loss/test', loss2.avg, epoch)           
            tf_writer.add_scalar('acc/test_v_top1', video_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top5', video_top5.avg, epoch)
            tf_writer.add_scalar('acc/test_f_top1', fusion_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_f_top5', fusion_top5.avg, epoch)
        elif args.do_KL:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('g_loss/test', loss1.avg, epoch)
            tf_writer.add_scalar('k_loss/test', loss2.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top1', video_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top5', video_top5.avg, epoch)
        else:
            tf_writer.add_scalar('g_loss/test', loss1.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top1', video_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_v_top5', video_top5.avg, epoch)
    if 'epic55' in args.dataset:
        return verb_top1.avg
    elif args.two_stream:
        return video_top1.avg, fusion_top1.avg
    else:
        return video_top1.avg