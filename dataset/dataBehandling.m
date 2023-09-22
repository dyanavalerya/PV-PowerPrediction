clc
clear 
close all

metadata = readtable("metadata.csv");
station00 = readmatrix("station00.csv");

station01 = readmatrix("station01.csv");
station02 = readmatrix("station02.csv");
station03 = readmatrix("station03.csv");
station04 = readmatrix("station04.csv");
station05 = readmatrix("station05.csv");
station06 = readmatrix("station06.csv");
station07 = readmatrix("station07.csv");
station08 = readmatrix("station08.csv");
station09 = readmatrix("station09.csv");

x1=[1:50000];
figure(1)
plot(x1(1:960),station00(1:960,15))
figure(2)
plot(x1(1:960),station01(1:960,15))
figure(3)
plot(x1(1:960),station02(1:960,15))
figure(4)
plot(x1(1:960),station03(1:960,15))
figure(5)
plot(x1(1:960),station04(1:960,15))
figure(6)
plot(x1(1:960),station05(1:960,15))
figure(7)
plot(x1(1:960),station06(1:960,15))
figure(8)
plot(x1(1:960),station07(1:960,15))
figure(9)
plot(x1(1:960),station08(1:960,15))
figure(10)
plot(x1(1:960),station09(1:960,15))

%%

s5_clr_data = readmatrix("McClear/s5_clr_data.xlsx");

s7_clr_data = readmatrix('McClear/s7_clr_data.xlsx');

s8_clr_data = readmatrix("McClear/s8_clr_data.xlsx");


x=[1:1546];

figure(1)
plot(x(1:501)*0.25,s5_clr_data(:,2));
title("S5 Irradiation on horizontal plane at the top of atmosphere (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(2)
plot(x(1:1536)*0.25,s7_clr_data(:,2));
title("S7 Irradiation on horizontal plane at the top of atmosphere (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(3)
plot(x(1:96)*0.25,s8_clr_data(:,2));
title("S8 Irradiation on horizontal plane at the top of atmosphere (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")
%%
figure(1)
plot(x(1:501)*0.25,s5_clr_data(:,3));
title("S5 Clear sky global irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(2)
plot(x(1:1536)*0.25,s7_clr_data(:,3));
title("S7 Clear sky global irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(3)
plot(x(1:96)*0.25,s8_clr_data(:,3));
title("S8 Clear sky global irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")
%%
figure(1)
plot(x(1:501)*0.25,s5_clr_data(:,4));
title("S5 Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(2)
plot(x(1:1536)*0.25,s7_clr_data(:,4));
title("S7 Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(3)
plot(x(1:96)*0.25,s8_clr_data(:,4));
title("S8 Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

%%
figure(1)
plot(x(1:501)*0.25,s5_clr_data(:,5));
title("S5 Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(2)
plot(x(1:1536)*0.25,s7_clr_data(:,5));
title("S7 Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(3)
plot(x(1:96)*0.25,s8_clr_data(:,5));
title("S8 Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

%%
figure(1)
plot(x(1:501)*0.25,s5_clr_data(:,6));
title("S5 Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(2)
plot(x(1:1536)*0.25,s7_clr_data(:,6));
title("S7 Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")

figure(3)
plot(x(1:96)*0.25,s8_clr_data(:,6));
title("S8 Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")
%%
close all
test=0;
for i = 1:1536
    test(i) = s7_clr_data(i,4) + s7_clr_data(i,5);
end
plot(x(1:1536)*0.25,test);
hold on
plot(x(1:1536)*0.25,s7_clr_data(:,3)+10);
