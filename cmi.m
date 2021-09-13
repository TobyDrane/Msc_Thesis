function c = cmi(x, y, z)
    assert(numel(x) == numel(y));
    n = numel(x);
    x = reshape(x,1,n);
    y = reshape(y,1,n);
    z = reshape(z,1,n);

    l = min(z);
    z = z - l + 1;
    
    for zIter=min(z):max(z)   
        indices = find(z==zIter);
        xSlice = x(indices);
        ySlice = y(indices);
        
        
        l = min(min(xSlice),min(ySlice));
        xSlice = xSlice-l+1;
        ySlice = ySlice-l+1;
        k = max(max(xSlice),max(ySlice));
       
        
        idx = 1:length(indices);
        Mx = sparse(idx,xSlice,1,length(indices),k,length(indices));
        My = sparse(idx,ySlice,1,length(indices),k,length(indices));
        Pxy = nonzeros(Mx'*My/length(indices)); %joint distribution of x and y
        Hxy = -dot(Pxy,log2(Pxy+eps));
        
        Px = mean(Mx,1);
        Py = mean(My,1);
        
        % entropy of Py and Px
        Hx = -dot(Px,log2(Px+eps));
        Hy = -dot(Py,log2(Py+eps));
        
        % mutual information for this value of z
        q(zIter) = Hx + Hy - Hxy;
        indexCount(zIter) = length(indices);
        
    end
    
    c = dot(q,indexCount/n);
end