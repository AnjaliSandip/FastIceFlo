load JKS8e4_wo
    figure(1),clf
        subplot(121)
        title('Ice Velocity_X, meters/year')
        patch( 'Faces',index, 'Vertices',[x y],'FaceVertexCData',ans.PTsolution.Vx,'FaceColor','interp','EdgeColor','None');
        colorbar
        subplot(122)
         title('Ice Velocity_Y')
        patch( 'Faces',index, 'Vertices',[x y],'FaceVertexCData',sqrt(ans.PTsolution.Vx.^2 + ans.PTsolution.Vy.^2)*yts,'FaceColor','interp','EdgeColor','None');
        colorbar
        drawnow