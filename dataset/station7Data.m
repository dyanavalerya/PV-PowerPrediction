clc
close all
clear all

station07 = readmatrix("station07.csv");
s7_clr_data = readmatrix('McClear/s7_clr_data.xlsx');


x=[1:32928];

figure(1)
plot(x(1:960)*0.25 + 16*4,station07(1:960,15))
title("S7 power over time")
xlabel("Time [h]")
ylabel("Power [MW]")

yyaxis left
plot(x(1:960)*0.25 + 16*4,station07(1:960,15));

yyaxis right
plot(x(1:960)*0.25 + 16*4,station07(1:960,2));




%%
subplot(5,1,1)
plot(x(1:960)*0.25,s7_clr_data(1:960,2));
title("S7 Irradiation on horizontal plane at the top of atmosphere (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")



subplot(5,1,2)
plot(x(1:960)*0.25,s7_clr_data(1:960,3));
title("S7 Clear sky global irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")


subplot(5,1,3)
plot(x(1:960)*0.25,s7_clr_data(1:960,4));
title("S7 Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")


subplot(5,1,4)
plot(x(1:960)*0.25,s7_clr_data(1:960,5));
title("S7 Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")


subplot(5,1,5)
plot(x(1:960)*0.25,s7_clr_data(1:960,6));
title("S7 Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)")
xlabel("Time [h]")
ylabel("Power [Wh/m2]")
