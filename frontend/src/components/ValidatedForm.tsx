import type { FormHTMLAttributes } from "react";

export default function ValidatedForm({
  onInvalidCapture,
  ...props
}: FormHTMLAttributes<HTMLFormElement>): React.ReactElement {
  return <form {...props} onInvalidCapture={(event) => {
    onInvalidCapture?.(event);
    const first = Array.from(event.currentTarget.elements).find((element) =>
      (element instanceof HTMLInputElement || element instanceof HTMLSelectElement || element instanceof HTMLTextAreaElement) && element.willValidate && !element.validity.valid,
    );
    if (first instanceof HTMLElement) first.focus();
  }} />;
}
