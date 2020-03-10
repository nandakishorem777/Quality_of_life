import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from srm_model import get_model
from srm_dataset import get_data

device = torch.device('cuda')
metrics = ['$', 'Water', 'Toilet', 'Roof', 'Cook', 'Drought', 'Pop', 'Livestock', 'Agri']


def loss_batch(model, criterion, xb, yb, opt=None):
    rgb, nir = xb[0].to(device), xb[1].to(device)
    # xb = xb.to(device)
    for k,v in yb.items():
        yb[k] = v.to(device)

    wealth, water_src, toilet_type, roof, cooking_fuel, drought, pop_density, livestock_bin, agriculture_land_bin = model((rgb, nir))

    wealth_loss = criterion(wealth, yb['wealth'])
    wealth_pred = torch.argmax(wealth, dim=1)
    wealth_acc  = (wealth_pred == yb['wealth']).sum()

    water_src_loss = criterion(water_src, yb['water_src'])
    water_src_pred = torch.argmax(water_src, dim=1)
    water_src_acc  = (water_src_pred == yb['water_src']).sum()

    toilet_type_loss = criterion(toilet_type, yb['toilet_type'])
    toilet_type_pred = torch.argmax(toilet_type, dim=1)
    toilet_type_acc  = (toilet_type_pred == yb['toilet_type']).sum()

    roof_loss = criterion(roof, yb['roof'])
    roof_pred = torch.argmax(roof, dim=1)
    roof_acc  = (roof_pred == yb['roof']).sum()

    cooking_fuel_loss = criterion(cooking_fuel, yb['cooking_fuel'])
    cooking_fuel_pred = torch.argmax(cooking_fuel, dim=1)
    cooking_fuel_acc  = (cooking_fuel_pred == yb['cooking_fuel']).sum()

    drought_loss = criterion(drought, yb['drought'])
    drought_pred = torch.argmax(drought, dim=1)
    drought_acc  = (drought_pred == yb['drought']).sum()

    pop_density_loss = criterion(pop_density, yb['pop_density'])
    pop_density_pred = torch.argmax(pop_density, dim=1)
    pop_density_acc  = (pop_density_pred == yb['pop_density']).sum()

    livestock_bin_loss = criterion(livestock_bin, yb['livestock_bin'])
    livestock_bin_pred = torch.argmax(livestock_bin, dim=1)
    livestock_bin_acc  = (livestock_bin_pred == yb['livestock_bin']).sum()

    agriculture_land_bin_loss = criterion(agriculture_land_bin, yb['agriculture_land_bin'])
    agriculture_land_bin_pred = torch.argmax(agriculture_land_bin, dim=1)
    agriculture_land_bin_acc  = (agriculture_land_bin_pred == yb['agriculture_land_bin']).sum()


    # print(wealth_acc.item()/xb.shape[0]*100, water_src_acc.item()/xb.shape[0]*100, toilet_type_acc.item()/xb.shape[0]*100, roof_acc.item()/xb.shape[0]*100)
    loss = 5 * wealth_loss + water_src_loss + toilet_type_loss + roof_loss + cooking_fuel_loss + drought_loss + pop_density_loss + livestock_bin_loss + agriculture_land_bin_loss

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    accs = [wealth_acc.item(), water_src_acc.item(), toilet_type_acc.item(), roof_acc.item(), cooking_fuel_acc.item(), drought_acc.item(), pop_density_acc.item(), livestock_bin_acc.item(), agriculture_land_bin_acc.item()]

    return (loss.item(), accs, xb[0].shape[0])


def fit(epochs, model, opt, scheduler, criterion, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for idx, (xb, yb) in enumerate(train_dl):
            loss, accs, num = loss_batch(model, criterion, xb, yb, opt)
            if idx % 10 == 0:
                print(f'Epoch: {epoch}, T.Loss: {loss:.3f}, Accs: {[(metrics[i], round(e/num*100, 2)) for i,e in enumerate(accs)]}')

        model.eval()
        with torch.no_grad():
            losses, accs, nums = .0, [.0]*9, .0
            for idx, (xb, yb) in enumerate(valid_dl):
                loss, acc, num = loss_batch(model, criterion, xb, yb)
                losses += loss
                nums += num
                for i in range(9):
                    accs[i] += acc[i]

        scheduler.step()
        print(f'Epoch: {epoch}, V.Loss: {(losses/len(valid_dl)):.3f}, Accs: {[(metrics[i], round(e/nums*100, 2)) for i,e in enumerate(accs)]}')


class Learner:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.wrapper.model = self.wrapper.model.to(device)

    def fit(self, epochs, lr, db, opt_fn):
        opt = opt_fn(self.wrapper.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CyclicLR(opt,
                                                mode='triangular2',
                                                base_lr=lr/10,
                                                max_lr=lr,
                                                step_size_up=int(len(db.train_dl)*0.3),
                                                step_size_down=int(len(db.train_dl)*0.7),
                                                cycle_momentum=False)
        criterion = F.cross_entropy
        fit(epochs, self.wrapper.model, opt, scheduler, criterion, db.train_dl, db.valid_dl)


def main():
    wrapper = get_model()
    db = get_data(img_sz=128, bs=64)
    opt_fn = optim.Adam

    learn = Learner(wrapper)
    learn.fit(2, 1e-1, db, opt_fn)

if __name__ == "__main__":
    main()
