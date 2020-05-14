function out = fun_dummy(y,t,Ret)
    out = y.*[Ret.a , Ret.b, Ret.c, Ret.d]-t;
end