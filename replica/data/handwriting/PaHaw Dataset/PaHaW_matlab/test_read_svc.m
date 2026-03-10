pth_svc = 'C:\Users\megha\OneDrive\Attachments\Desktop\Homework\Projects in BME AI\PaHaW\PaHaW_matlab\00001__1_1.svc';
Y = read_SVC_file_2(pth_svc, 1);

figure;
plot(Y(:,1), Y(:,2));
axis equal;
title('Handwriting Trajectory');