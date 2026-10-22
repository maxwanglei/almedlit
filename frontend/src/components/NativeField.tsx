import { useCallback, useId, useLayoutEffect, useRef, useState, type ComponentPropsWithRef, type DOMAttributes, type Ref } from "react";

type FormControl = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
type ValidationProps<T extends FormControl> = Pick<DOMAttributes<T>, "onInvalid" | "onBlur" | "onInput"> & Pick<ComponentPropsWithRef<"input">,
  "id" | "aria-invalid" | "aria-describedby" | "value" | "checked" | "required" | "disabled" | "pattern" | "min" | "max" | "minLength" | "maxLength" | "type" | "multiple"
> & {
  ref?: Ref<T>;
};

function useFieldValidation<T extends FormControl>(props: ValidationProps<T>) {
  const generatedId = useId();
  const id = props.id ?? generatedId;
  const [message, setMessage] = useState("");
  const hasCustomError = (props["aria-invalid"] === true || props["aria-invalid"] === "true") && Boolean(props["aria-describedby"]);
  const visibleMessage = hasCustomError ? "" : message;
  const controlRef = useRef<T | null>(null);
  const forwardedRef = props.ref;
  const ref = useCallback((element: T | null) => {
    controlRef.current = element;
    if (typeof forwardedRef === "function") return forwardedRef(element);
    if (forwardedRef) forwardedRef.current = element;
  }, [forwardedRef]);
  useLayoutEffect(() => {
    const control = controlRef.current;
    if (!message || !control) return;
    const currentMessage = hasCustomError || !control.willValidate || control.validity.valid ? "" : control.validationMessage;
    if (currentMessage !== message) setMessage(currentMessage);
  }, [hasCustomError, message, props.value, props.checked, props.required, props.disabled, props.pattern, props.min, props.max, props.minLength, props.maxLength, props.type, props.multiple]);
  const errorId = `${id}-error`;
  return {
    field: {
      id,
      ref,
      "aria-invalid": visibleMessage ? true : props["aria-invalid"],
      "aria-describedby": [props["aria-describedby"], visibleMessage ? errorId : null].filter(Boolean).join(" ") || undefined,
      onInvalid: (event: React.InvalidEvent<T>) => {
        props.onInvalid?.(event);
        event.preventDefault();
        setMessage(event.currentTarget.validationMessage);
      },
      onBlur: (event: React.FocusEvent<T>) => {
        props.onBlur?.(event);
        setMessage(event.currentTarget.validity.valid ? "" : event.currentTarget.validationMessage);
      },
      onInput: (event: React.InputEvent<T>) => {
        props.onInput?.(event);
        if (message) setMessage(event.currentTarget.validity.valid ? "" : event.currentTarget.validationMessage);
      },
    },
    error: visibleMessage ? <span id={errorId} className="platform-field-error" aria-hidden="true">{visibleMessage}</span> : null,
  };
}

export function NativeInput(props: ComponentPropsWithRef<"input">): React.ReactElement {
  const { field, error } = useFieldValidation<HTMLInputElement>(props);
  return <><input {...props} {...field} />{error}</>;
}

export function NativeSelect(props: ComponentPropsWithRef<"select">): React.ReactElement {
  const { field, error } = useFieldValidation<HTMLSelectElement>(props);
  return <><select {...props} {...field} />{error}</>;
}

export function NativeTextarea(props: ComponentPropsWithRef<"textarea">): React.ReactElement {
  const { field, error } = useFieldValidation<HTMLTextAreaElement>(props);
  return <><textarea {...props} {...field} />{error}</>;
}
